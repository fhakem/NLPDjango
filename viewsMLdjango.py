import os
import zipfile
import time
import boto3
import json
import datetime
import logging

from django.http import JsonResponse, HttpResponse
from django.conf import settings
from rest_framework.decorators import api_view
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema

from .core.IntentAndEntityModel import IntentAndEntityModel
from .core.standard_entities import detect_standard_entities, parse_entities


logger = logging.getLogger("django")

models = {}  # The loaded models are stored in this dictionary


def load_intent_and_entity_model(path, name):
    """
    Load a model from a directory and store it in the model dictionary.

    :param str path: path to the model directory
    :param str name: target key of the model in the model dictionary
    """

    assert name not in models
    model = IntentAndEntityModel(os.path.join(settings.MEDIA_ROOT, "models/{}".format(path)))
    models[name] = model

    logger.info("Successfully loaded model '{}'.".format(name))


def load_model_from_s3(s3_bucket_name, file_key, name):
    """
    Load a model stored as a ZIP file on an S3 bucket.

    :param str s3_bucket_name: S3 bucket name
    :param str file_key: file key of the ZIP file in the S3 bucket
    :param str name: target key of the model in the model dictionary
    :return:
    """
    file_name = "remote_" + file_key.replace("\\", "/").split("/")[-1]
    zip_file_path = os.path.join(settings.MEDIA_ROOT, "models/{}".format(file_name))

    s3 = boto3.resource('s3')
    s3.meta.client.download_file(s3_bucket_name, file_key, zip_file_path)

    directory_path = zip_file_path[:-4]
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(directory_path)
    os.remove(zip_file_path)

    load_intent_and_entity_model(file_name[:-4], name)


# Try to load the models from the S3 bucket. If not allowed then load models from the media files
try:
    model_config_json_path = os.path.join(settings.BASE_DIR, "model_config.json")
    with open(model_config_json_path) as json_file:
        model_config_dict = json.load(json_file)
        for model_name_, values in model_config_dict.items():
            bucket_name = values["bucket_name"]
            model_key = values["model_key"]
            load_model_from_s3(bucket_name, model_key, model_name_)

except Exception as err:
    print("Error loading remote model: ", err)

    load_intent_and_entity_model(path="intent_and_entity_model", name="intent_and_entity_model")
    load_intent_and_entity_model(path="intent_and_entity_model_sm", name="intent_and_entity_model_small_talk")
    load_intent_and_entity_model(path="rf", name="rf")


print(models)


def check_request_data(key, request_data, default=None):
    """
    Check if a parameter is provided in the input request data.

    :param str key: name of the parameter to check
    :param dict[str, Any] request_data: input request data
    :param Any default: if provided, then the value to return if the parameter is not in the request data
    :return: value of the parameter in the request data
    """
    if key in request_data:
        return request_data[key]
    elif default is not None:
        return default
    else:
        raise ValueError("The parameter '{}' has not been provided.".format(key))


def tokenize_raw_sentence(raw_sentence, tokenized_sentence):
    """
    Tokenize a raw sentence to match the tokenized sentence
    (i.e. each token from the tokenized sentence has its correspondent in the tokenized raw sentence).

    :param str raw_sentence: raw sentence to tokenize
    :param list[str] tokenized_sentence: list of sentence tokens
    :return: tokenized raw sentence
    """
    raw_sentence = " ".join(raw_sentence.split())
    sentence = " ".join(raw_sentence.lower().split())
    tokenized_raw_sentence = []
    for token in tokenized_sentence:
        if token == "":
            tokenized_raw_sentence.append(token)
        else:
            try:
                token_index = sentence.index(token)
                tokenized_raw_sentence.append(raw_sentence[token_index:token_index + len(token)])
            except ValueError as err_msg:
                if str(err_msg) == "substring not found":
                    tokenized_raw_sentence.append("")
                else:
                    raise
    return tokenized_raw_sentence


@api_view(["GET"])
def health(request):

    if request.method == "GET":

        if not settings.PERMISSION_CLASS().has_permission(request, None):
            return JsonResponse(
                {"message": "IP address not allowed."}, status=403)

        return JsonResponse({"message": "Application live."}, status=200)


@swagger_auto_schema(
    method='POST',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            "text": openapi.Schema(description="The sentence to classify.", type=openapi.TYPE_STRING),
            "model_name": openapi.Schema(
                description="The name of the model to use for intent classification and entity detection.",
                type=openapi.TYPE_STRING, default="intent_and_entity_model"),
            "intent_filter": openapi.Schema(
                description="The intents to detect in the text. It could be a list of intents separated by commas "
                            "(e.g. 'small-oui-nt0151, small-non-nt0149'), an intent family root (e.g. 'small'), "
                            "or '*' or 'all' for no filter.", type=openapi.TYPE_STRING, default="*"),
            "k": openapi.Schema(
                description="The number of top intents to return.", type=openapi.TYPE_NUMBER, default=1),
        }
    ),
    responses={
        "200": openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "results": openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        "intents": openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            items=openapi.Schema(
                                type=openapi.TYPE_OBJECT,
                                properties={
                                    "slug": openapi.Schema(
                                        type=openapi.TYPE_STRING, description="The name of the detected intent"),
                                    "confidence": openapi.Schema(
                                        type=openapi.TYPE_NUMBER, description="The score of the detected intent")
                                }
                            )
                        ),
                        "entities": openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                "#entity_name": openapi.Schema(
                                    type=openapi.TYPE_ARRAY,
                                    items=openapi.Schema(
                                        type=openapi.TYPE_OBJECT,
                                        properties={
                                            "raw": openapi.Schema(type=openapi.TYPE_STRING),
                                            "confidence": openapi.Schema(type=openapi.TYPE_NUMBER),
                                            "value": openapi.Schema(type=openapi.TYPE_STRING),
                                        }
                                    )
                                )
                            }
                        )
                    }
                ),
                "computation_time": openapi.Schema(
                    type=openapi.TYPE_STRING, description="The time (in seconds) between the request "
                                                          "being received and the response being sent."),
            }
        ),
        "400": openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "message": openapi.Schema(description="The error message.", type=openapi.TYPE_STRING)
            }
        ),
        "403": openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "message": openapi.Schema(description="The authentication error message.", type=openapi.TYPE_STRING)
            }
        )
    }
)
@api_view(["POST"])
def predict(request):
    """
    Predict the intents and entities from an input request.
    """

    if request.method == "POST":

        if not settings.PERMISSION_CLASS().has_permission(request, None):
            return JsonResponse(
                {"logs": ["IP address not allowed."], "status": 403}, status=403)

        raw_sentence = None
        model_name = None

        try:
            start_time = time.time()

            request_data = request.data
            if "text" in request_data:
                sentence = check_request_data("text", request_data)
            else:
                sentence = check_request_data("message", request_data)["content"]
            model_name = check_request_data("model_name", request_data, default="intent_and_entity_model")
            intent_filter = check_request_data("intent_filter", request_data, default="*")
            k = int(check_request_data("k", request_data, default=1))

            if model_name not in models:
                error_message = "The model '{}' has not been loaded yet.".format(model_name)
                logger.error("ModelNotLoadedError: {}".format(error_message))
                return JsonResponse({"logs": [error_message], "status": 400}, status=400)

            intent_and_entity_model = models[model_name]

            raw_sentence = sentence
            sentence = " ".join(sentence.lower().split())  # Put in lowercase and remove multiple spaces
            tokenized_sentence = intent_and_entity_model.split_sentences([sentence])[0]
            tokenized_raw_sentence = tokenize_raw_sentence(raw_sentence, tokenized_sentence)

            predictions = intent_and_entity_model.predict([sentence], k=k, intent_filter=intent_filter)

            intent_predictions = predictions[0][0]["intent"]
            entity_predictions = predictions[0][1:]

            intents = [{
                "slug": intent_predictions["labels"][i],
                "confidence": float(intent_predictions["scores"][i]),
            } for i in range(len(intent_predictions["scores"]))]

            standard_entities = detect_standard_entities(sentence)
            entities = parse_entities(sentence, tokenized_raw_sentence, entity_predictions,
                                      standard_entities=standard_entities)

            response = {
                "results": {
                    "intents": intents,
                    "entities": entities
                },
                "logs": [],
                "timestamp": str(datetime.datetime.utcnow().isoformat()),
                "status": 200,
                "source": raw_sentence,
                "computation_time": time.time() - start_time,
            }

            logger.info("Prediction - input: '{}', model: '{}', intent: '{}', entities: {}".format(
                raw_sentence, model_name, intents[0]["slug"], list(entities.keys())))

            return JsonResponse(response, status=200)

        except Exception as error:
            if raw_sentence is not None and model_name is not None:
                logger.info("Prediction - input: '{}', model: '{}'".format(
                    raw_sentence, model_name))
            logger.error(error)
            return JsonResponse({"logs": [str(error)], "status": 400}, status=400)


@api_view(["GET"])
def label_list(request):
    """
    Give the label_list used by the default model
    """

    if request.method == "GET":

        if not settings.PERMISSION_CLASS().has_permission(request, None):
            return JsonResponse(
                {"message": "IP address not allowed."}, status=403)

        try:
            model = models["intent_and_entity_model"]
            labels = model.label_list
            neutral_token_index = labels.index("O")
            intents_list = labels[:neutral_token_index]
            entities_list = labels[neutral_token_index:]
            response = {
                "intents_list": intents_list,
                "entities_list": entities_list
            }
            return JsonResponse(response, status=200)
        
        except Exception as error:
            return JsonResponse({"message": str(error)}, status=400)

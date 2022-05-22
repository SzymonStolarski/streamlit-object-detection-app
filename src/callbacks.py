import base64
import json
import requests
import time

from src.api_configuration import API_PATH


def detect_objects_callback(uploaded_images,
                            min_score,
                            model_name,
                            filtered_predictions) -> dict:

    dict_with_responses = {}
    img_iterator = 1
    for img in uploaded_images:
        img_bytes = img.read()
        img_str = base64.b64encode(img_bytes).decode("utf-8")
        json_input = json.dumps({
            'img_base64': img_str,
            'min_score': min_score,
            'model_name': model_name,
            'filter_predictions': filtered_predictions
        })

        post_response = requests.post(
            f"{API_PATH}/predict_static_img/", json_input).json()

        # Get request every 2 seconds to check if the prediction
        # of image has finished/failed. If 10 requests are reached
        # then raise TimeoutError
        get_iterator = 0
        while True:
            get_status_response = requests.get(
                f"{API_PATH}/prediction_statuses/"
                f"{post_response['prediction_id']}").json()
            get_iterator += 1
            statuses = [v['status'] for v
                        in list(get_status_response.values())]

            if 'finished' in statuses:
                get_prediction_response = requests.get(
                    f"{API_PATH}/predictions/{post_response['prediction_id']}"
                                                       ).json()
                dict_with_responses[
                    f'img_{img_iterator}'] = get_prediction_response
                break

            elif 'error' in statuses:
                raise SystemError('An error has occured during detection')

            time.sleep(2)
            if get_iterator == 10:
                raise TimeoutError('Timeout limit reached')

        img_iterator += 1

    return dict_with_responses

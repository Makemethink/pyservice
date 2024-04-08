import json
from django.http import HttpResponse, JsonResponse
from . import recommender as recom


# Create your views here.
def get_suggestions(request):
    input_data = {}
    try:
        input_data = json.loads(request.body)
    except json.decoder.JSONDecodeError as e:
        print(f'Exception occurred : {e}')

    hotel_menu = input_data.get('availableProducts')
    ordered_items = input_data.get('orderedProducts')
    recommending_item_count = input_data.get('suggestionCount')

    if hotel_menu is None or ordered_items is None:
        return JsonResponse({})

    product_id = recom.run(hotel_menu, ordered_items, recommending_item_count)

    return JsonResponse(product_id)

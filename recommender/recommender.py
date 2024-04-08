from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# The core process of content based filtration happens here
def content_based_filtering(user_items, menu_items, model, num_recommendations: int=5) -> set:
    """ Step 1 : Filtering out-of-vocabularies and vectorization
        Step 2 : Cosine Similarity """
    recommendations = set()

    try:
        if not user_items or not menu_items:
            return recommendations

        # Filter out-of-vocabulary words
        user_vectors = np.array([model.wv[word] for word in user_items if word in model.wv.key_to_index])
        menu_vectors = np.array([model.wv[word] for word in menu_items if word in model.wv.key_to_index])

        # If no vectors available please return
        if len(user_vectors) == 0 or len(menu_vectors) == 0:
            return recommendations

        # Find the cosine similarity
        similarity_scores = cosine_similarity(user_vectors, menu_vectors)

        # Find indices of top similar items for each user item
        top_indices = similarity_scores.argsort(axis=1)[:, ::-1]

        # Generate recommendations based on top similar items
        for indices in top_indices:
            for idx in indices:
                if menu_items[idx] not in user_items:
                    recommendations.add(menu_items[idx])
                    if len(recommendations) == num_recommendations:
                        return recommendations

    except Exception as ex:
        print('Exception occurred : ', ex)

    return recommendations


def get_id_by_name(products: dict, name: str) -> int:
    try:
        for product in products:
            if product["name"] == name:
                return product["id"]
    except Exception as ex:
        print('Exception occurred : ', ex)
    return -1


# Program entry point
def run(hotel_menu: dict, user_ordered_items: dict, suggestion_count: int) -> dict:

    food_id = []

    try:
        available_items = []
        for dict_item in hotel_menu:
            available_items.append(dict_item.get('name'))

        ordered_items = []
        for dict_item in user_ordered_items:
            ordered_items.append(dict_item.get('name'))

        if suggestion_count > (len(available_items) - len(ordered_items)):
            return {'productIds': food_id}

        # In the future, we can train a model specifically for south indian food items
        # And we can use our own model here ..
        word2vec_model = Word2Vec(sentences=[available_items], vector_size=100, window=7, min_count=1, workers=4)

        # Content based filtering happens here
        suggestions = list(content_based_filtering(ordered_items, available_items, word2vec_model, suggestion_count))

        for item in suggestions:
            product_id = get_id_by_name(hotel_menu, item)
            if product_id != -1:
                food_id.append(product_id)

    except Exception as ex:
        print('Exception occurred', ex)

    # Changing the set to list, in order to avoid TypeError: Object of type set is not JSON serializable
    return {'productIds': food_id}

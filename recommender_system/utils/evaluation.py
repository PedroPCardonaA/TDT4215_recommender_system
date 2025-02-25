from typing import Any, Dict
import os
from codecarbon import EmissionsTracker
from typing import Callable, Any, Tuple
import polars as pl
import numpy as np


def perform_model_evaluation(model: Any, test_data: pl.DataFrame, k: int = 5) -> dict:
    """
    Evaluate a recommender model using precision, recall, and FPR at k.

    For each user in the test set, relevant items are defined as the set of article_ids
    that the user has in the test data. The model's recommendations (obtained via either 
    model.recommend(user_id, n=k) or model.recommend_n_articles(user_id, n=k)) are then compared
    against these relevant items. The candidate set for negatives is defined as all unique article_ids
    in test_data.

    Parameters
    ----------
    model : Any
        A recommender model with a recommend(user_id, n) or recommend_n_articles(user_id, n) method.
    test_data : pl.DataFrame
        A DataFrame containing test interactions (must include "user_id" and "article_id" columns).
    k : int, optional
        Number of top recommendations to consider. Default is 5.

    Returns
    -------
    dict
        A dictionary with average precision@k, recall@k, and FPR@k.
    """
    # Determine which recommendation method to use.
    if hasattr(model, "recommend"):
        rec_func = model.recommend
    elif hasattr(model, "recommend_n_articles"):
        rec_func = model.recommend_n_articles
    else:
        raise ValueError("Model must have a 'recommend' or 'recommend_n_articles' method.")

    # Candidate set: all unique article IDs in test_data.
    candidate_set = set(test_data.select("article_id").unique().to_numpy().flatten())
    
    # Get unique users from test data.
    user_ids = test_data.select("user_id").unique().to_numpy().flatten()
    precisions = []
    recalls = []
    fprs = []
    
    for user in user_ids:
        # Relevant items: all article_ids for this user in test_data.
        user_test = test_data.filter(pl.col("user_id") == user)
        relevant_items = set(user_test.select("article_id").to_numpy().flatten())
        if not relevant_items:
            continue

        # Get recommendations for this user using the determined recommendation function.
        recommended_items = rec_func(user, n=k)
        
        # Compute hits.
        hits = sum(1 for item in recommended_items if item in relevant_items)
        precision = hits / k
        recall = hits / len(relevant_items)
        
        # Compute FPR:
        negatives = candidate_set - relevant_items
        false_positives = sum(1 for item in recommended_items if item not in relevant_items)
        fpr = false_positives / len(negatives) if negatives else 0.0

        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fpr)

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_fpr = np.mean(fprs) if fprs else 0.0
    
    return {"precision@k": avg_precision, "recall@k": avg_recall, "fpr@k": avg_fpr}

def record_carbon_footprint(function_name: str , model_name: str, /, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Execute the provided function while tracking its carbon footprint and save the emissions data
    to output/<function_name>_emission.csv.

    This utility method creates an "output" directory if it doesn't exist, then initializes an
    EmissionsTracker configured to write its output to "output/<function_name>_emission.csv". It runs
    the given function, stops the tracker, and returns a tuple containing the function's result and
    the recorded emissions (in kgCO2e).

    Parameters
    ----------
    function_name : str
        A name for the function being tracked; used to generate the output file name.
        This parameter is positional-only.
    func : callable
        The function to execute.
    *args :
        Positional arguments to pass to the function.
    **kwargs :
        Keyword arguments to pass to the function.

    Returns
    -------
    tuple
        A tuple (result, emissions) where result is the output of the function and emissions is
        the estimated carbon footprint in kgCO2e.
    """
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ## Adds model name and function name to the output file name path
    output_filename = f"{model_name}_{function_name}_emission.csv"
    tracker = EmissionsTracker(output_dir=output_dir, output_file=output_filename)
    tracker.start()
    result = func(*args, **kwargs)
    emissions = tracker.stop()
    return result, emissions



def track_model_energy(model: Any, model_name: str, user_id: int, n: int = 5) -> Dict[str, tuple]:
    """
    Track the carbon footprint of the model's .fit() and .recommend() methods.

    This utility method calls model.fit() and model.recommend(user_id, n=n) (or model.recommend_n_articles(user_id, n=n))
    while tracking their energy consumption via the record_carbon_footprint function. It returns a dictionary
    with the results and emissions for each method call.

    Parameters
    ----------
    model : Any
        A recommender model with .fit() and either a recommend(user_id, n) or recommend_n_articles(user_id, n) method.
    user_id : int
        The user ID for which to obtain recommendations.
    n : int, optional
        The number of recommendations to return (default is 5).

    Returns
    -------
    dict
        A dictionary with keys "fit" and "recommend" where each value is a tuple (result, emissions)
        corresponding to the respective method call.
    """
    # Track the energy consumption of the .fit() method.
    fit_result, fit_emissions = record_carbon_footprint("fit", model_name, model.fit)
    
    # Determine which recommendation method to use.
    if hasattr(model, "recommend"):
        rec_func = model.recommend
    elif hasattr(model, "recommend_n_articles"):
        rec_func = model.recommend_n_articles
    else:
        raise ValueError("Model must have a 'recommend' or 'recommend_n_articles' method.")
    
    # Track the energy consumption of the recommendation method.
    recommend_result, recommend_emissions = record_carbon_footprint("recommend", model_name, rec_func, user_id, n=n)
    
    return {
        "fit": (fit_result, fit_emissions),
        "recommend": (recommend_result, recommend_emissions)
    }

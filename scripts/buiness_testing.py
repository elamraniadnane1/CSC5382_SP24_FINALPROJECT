from sklearn.metrics import classification_report
//To be added to the previous pipeline
@step
def business_testing(model, test_data: pd.DataFrame, business_params: dict):
    """
    Evaluate the model based on business-centric and performance metrics.

    Args:
        model: The trained model.
        test_data (pd.DataFrame): Testing data containing features and true labels.
        business_params (dict): Parameters such as costs associated with false positives and negatives.

    Returns:
        dict: A dictionary containing calculated business and performance metrics.
    """
    business_metrics = {
        'total_cost': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'performance_metrics': {}
    }

    # Apply model to test data to get predictions
    predictions = model.predict(test_data.drop(['true_label'], axis=1))

    # Calculate business metrics
    for true_label, prediction in zip(test_data['true_label'], predictions):
        if true_label != prediction:
            if prediction == 'positive_class':
                business_metrics['false_positives'] += 1
                business_metrics['total_cost'] += business_params['cost_false_positive']
            else:
                business_metrics['false_negatives'] += 1
                business_metrics['total_cost'] += business_params['cost_false_negative']

    # Compute performance metrics
    report = classification_report(test_data['true_label'], predictions, output_dict=True)
    performance_metrics = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    business_metrics['performance_metrics'] = performance_metrics

    # Check if all metrics are above 70%
    metrics_pass = all(value >= 0.70 for value in performance_metrics.values())
    business_metrics['metrics_pass'] = metrics_pass

    return business_metrics

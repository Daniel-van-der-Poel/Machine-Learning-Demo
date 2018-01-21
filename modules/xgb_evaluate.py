import numpy as np
import pandas as pd


def xgb_evaluate(preds, truths, features, params, sets, version, iteration):

    #  Create DataFrame with all the required data
    result = pd.DataFrame(preds, columns=['prediction']).round().astype(int)
    result['truth'] = truths
    result['correct'] = (result['prediction'] == result['truth']).astype(int)
    result['true_positive'] = result['prediction'] * result['correct']
    result['true_negative'] = (1 - result['prediction']) * result['correct']
    result['false_positive'] = (1 - result['correct']) * result['prediction']
    result['false_negative'] = (1 - result['correct']) * (1 - result['prediction'])

    #  Calculate evaluation data
    n_samples = result.shape[0]
    n_correct_preds = result['correct'].sum()
    n_incorrect_preds = n_samples - n_correct_preds
    accuracy = n_correct_preds / n_samples
    result['true_positive'].sum()
    precision = (result['true_positive'].sum() / result['prediction'].sum()) if result['prediction'].sum() else 1
    recall = result['true_positive'].sum() / (result['true_positive'].sum() + result['false_negative'].sum())
    f1 = 2 * precision * recall / (precision + recall)
    perc_false_positives_all_incorrect = result['false_positive'].sum() / n_incorrect_preds
    perc_false_negatives_all_incorrect = result['false_negative'].sum() / n_incorrect_preds
    perc_true_positives_actual_positives = result['true_positive'].sum() / result['prediction'].sum()
    perc_false_positives_actual_positives = result['false_positive'].sum() / result['prediction'].sum()
    perc_true_positives_predicted_positives = result['true_positive'].sum() / result['truth'].sum()
    perc_false_negatives_predicted_positives = result['false_negative'].sum() / result['truth'].sum()

    #  Create evaluation statistics string
    eval_str = 'LendingClub model evaluation'
    eval_str +=  f'\nVersion: {version}, iteration: {iteration}'
    eval_str += '\n\nAccuracy: {:.3%}, F1 score: {:.3%}'.format(accuracy, f1)
    eval_str += '\nIncorrect predictions: {:.1%} false positives, {:.1%} false negatives'.format(
        perc_false_positives_all_incorrect, perc_false_negatives_all_incorrect)
    eval_str += '\nPredicted positives: {:.1%} true positives, {:.1%} false positives'.format(
        perc_true_positives_actual_positives, perc_false_positives_actual_positives)
    eval_str += '\nActual positives: {:.1%} true positives, {:.1%} false negatives'.format(
        perc_true_positives_predicted_positives, perc_false_negatives_predicted_positives)

    #  Add most important features
    eval_str += '\n\nMost important features:'
    important = features['ranked'][:10]
    for n in range(len(important)):
        eval_str += f'\n{n + 1}: {important[n]}'

    #  Add unimportant features
    n_unimportant = len(features['unimportant'])
    eval_str += f'\n\nUnimportant features: {n_unimportant}'

    #  Add datasets
    eval_str += f'\n\nDataset{"s" if len(sets) > 1 else ""}: '
    for key in sets:
        eval_str += f'{key} '

    #  Add parameters
    eval_str += '\n\nModel parameters:'
    for key in params:
        eval_str += f'\n{key}: {params[key]}'

    evaluation = {}
    evaluation['df'] = result
    evaluation['accuracy'] = accuracy
    evaluation['f1'] = f1
    evaluation['string'] = eval_str

    return evaluation

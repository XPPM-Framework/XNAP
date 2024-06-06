import os
import json
from pathlib import Path
import functools
import multiprocessing
import gc
from typing import Tuple, List

from pandas import DataFrame

import config as config
import utils as utils
from explanation.LSTM.LSTM_bidi import *
from explanation.util.heatmap import html_heatmap
import explanation.util.browser as browser
from nap.preprocessor import Preprocessor as Preprocessor
import nap.tester as test
import nap.trainer as train
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


# Disable tf warnings here as they are spammed by this function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def explain_trace(trace: DataFrame, log_params, preprocessor, args) -> DataFrame:
    process_instance = [*trace[log_params.get("activity_key", "event")].to_list(), "!"]
    #process_instance = preprocessor.get_random_process_instance(args.rand_lower_bound, args.rand_upper_bound)

    # Skip prediction and explanation for first event
    predictions = [None]
    ground_truths = [None]
    temporal_explanations = [[]]
    for prefix_index in range(2, len(process_instance)):
        # next activity prediction
        predicted_act_class, predicted_act_class_str, target_act_class, target_act_class_str, prefix_words, model, input_encoded, prob_dist = test.test_prefix(
            args, preprocessor, process_instance, prefix_index, model_path=args.model_path)
        #print("Prefix: %s; Next activity prediction: %s; Next activity target: %s" % (prefix_index, predicted_act_class, target_act_class_str))
        #print("Probability Distribution:")
        #print(prob_dist)

        # compute lrp relevances
        eps = 0.001  # small positive number
        bias_factor = 0.0  # recommended value
        net = LSTM_bidi(args, model, input_encoded)  # load trained LSTM model

        Rx, Rx_rev, R_rest = net.lrp(prefix_words, predicted_act_class, eps, bias_factor)  # perform LRP
        R_words = np.sum(Rx + Rx_rev, axis=1)  # compute word-level LRP relevances
        #scores = net.s.copy()  # classification prediction scores
        #print(R_words)
        #predictions.append(predicted_act_class_str)
        #temporal_explanations.append(list(R_words))
        predictions.append(predicted_act_class_str)
        ground_truths.append(target_act_class_str)
        temporal_explanations.append(list(R_words))

    trace["prediction"] = predictions
    trace["ground_truth"] = ground_truths
    trace["explanation"] = temporal_explanations
    # Force a garbage collection because of a Tensorflow memory leak problem
    gc.collect()

    return trace

    #prefix_heatmaps = prefix_heatmaps + html_heatmap(prefix_words, R_words) + "<br>"  # create heatmap
    #browser.display_html(prefix_heatmaps)  # display heatmap


def explain_log(event_log: DataFrame, log_params: dict, preprocessor, args, queue=None):
    predictions = []
    temporal_explanations = []
    activity_relevance_explanations = []
    cases = event_log.groupby(log_params.get("case_id_key", "case"))

    # Example usage:
    # Assuming df is your DataFrame and group_key is the column you want to group by

    # Initialize multiprocessing pool
    pool = multiprocessing.Pool()

    # Apply the process_groups function to each group using multiprocessing
    partial_func = functools.partial(explain_trace, log_params=log_params, preprocessor=preprocessor, args=args)
    processed_groups = pool.map(partial_func, [group for name, group in cases])

    # Close the pool
    pool.close()
    pool.join()

    # Concatenate the processed groups back into a single DataFrame
    final_df = pd.concat(processed_groups)

    # for case_id, trace in tqdm(cases, desc="Explain traces"):
    #     print(f"Explaining trace {case_id}")
    #     prediction, temporal_explanation = explain_trace(trace, log_params, preprocessor, args)
    #     predictions.append(prediction)
    #     temporal_explanations.append(temporal_explanation)
    #
    # event_log["prediction"] = predictions
    # event_log["explanation"] = temporal_explanations
    if queue is None:
        return final_df
    else:
        queue.put(final_df)


def main():
    args = config.load()
    output = utils.load_output()
    utils.clear_measurement_file(args)
    log_params: dict = json.loads(args.log_params)

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    settings_path = Path(args.model_path).parent / f"settings.json" \
        if args.model_path else Path(args.model_dir) / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # explanation mode for nap
    if args.explain:
        if settings_path.exists():
            settings = json.loads(settings_path.read_text())
            preprocessor = Preprocessor(settings, log_params)
            print(f"Read settings from {settings_path.resolve()}")
        else:
            preprocessor = Preprocessor(args, log_params)
            print(f"Took settings from args because not found in {settings_path}")

        #event_log = pd.read_csv(args.data_dir + args.data_set, sep=";", quotechar='|')
        event_log = pd.read_csv(args.data_dir + args.data_set)
        event_log.sort_values(by=[log_params.get("case_id_key", "case"), log_params.get("timestamp_key", 'time')],
                              inplace=True)

        unique_activities = event_log[log_params.get("activity_key", "event")].unique()

        if int(args.log_limit) > 0:
            print(f"Limiting log to {args.log_limit} entries")
            event_log = event_log[:int(args.log_limit)]

        #prefix_heatmaps: str = ""
        processed_chunks = []
        queue = multiprocessing.Queue()
        splits = split_on_case(event_log, 30, **log_params)
        for i, split in tqdm(enumerate(splits), ):
            print(f"Evaluating log split: {i}")
            # p = multiprocessing.Process(target=explain_log, args=(split, log_params, preprocessor, args, queue))
            # p.start()
            # p.join()
            # explained_log_chunk = queue.get()
            explained_log_chunk = explain_log(split, log_params, preprocessor, args)
            processed_chunks.append(explained_log_chunk)

        explained_log = pd.concat(processed_chunks)
        print("Explained log columns: ", explained_log.columns)
        explanation_path = Path(args.task) / args.result_dir / "local_explanations.csv"
        explanation_path.parent.mkdir(parents=True, exist_ok=True)
        explained_log.to_csv(explanation_path, index=False)
        print(f"Saved explanations to {explanation_path.resolve()}")

    # validation mode for nap
    elif not args.explain:
        preprocessor = Preprocessor(args, log_params)
        with open(settings_path, "w") as settings_file:
            json.dump(preprocessor.get_pure_dict(), settings_file)
            print(f"Saved settings to {settings_path.resolve()}")

        if args.cross_validation:
            for iteration_cross_validation in range(0, args.num_folds):
                preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

                output["training_time_seconds"].append(train.train(args, preprocessor))
                test.test(args, preprocessor)

                output = utils.get_output(args, preprocessor, output)
                utils.print_output(args, output, iteration_cross_validation)
                utils.write_output(args, output, iteration_cross_validation, args.result_dir)

            utils.print_output(args, output, iteration_cross_validation + 1)
            utils.write_output(args, output, iteration_cross_validation + 1, args.result_dir)

        else:
            output["training_time_seconds"].append(train.train(args, preprocessor))
            test.test(args, preprocessor)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, -1)
            utils.write_output(args, output, -1, args.result_dir)

    else:
        print("No mode selected ...")


# Copied from my framework to mitigate issues with memory leaks
def split_on_case(df: DataFrame, split_count: int, *,
                  case_id_key: str = 'case', timestamp_key: str = 'start_timestamp',
                  **kwargs
                  ) -> List[DataFrame]:
    """
    Split the dataset into a train and test dataset, preserving the case structure.
    :param df: The dataframe to split.
    :param split_count: The number of splits.
    :param case_id_key: The column name of the case id.
    :param timestamp_key: The column name of the timestamp to use for ordering the events.
    :return: A tuple containing the train and test dataframes.
    """
    case_ids = df[case_id_key].unique()
    cases = df.sort_values([case_id_key, timestamp_key], ascending=(True, True)).groupby(case_id_key)
    split_size = len(case_ids) // split_count
    split_remainder = len(case_ids) % split_count
    split_sizes = [split_size] * split_count
    for i in range(split_remainder):
        split_sizes[i] += 1
    splits = []
    for split_idx in range(0, split_count):
        splits.append(pd.concat([cases.get_group(case_id) for case_id in case_ids[:split_sizes[split_idx]]]))

    # select cases for train and test
    return splits


if __name__ == '__main__':
    main()

import json
from pathlib import Path

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
from tqdm import tqdm

if __name__ == '__main__':
    args = config.load()
    output = utils.load_output()
    utils.clear_measurement_file(args)
    log_params: dict = json.loads(args.log_params)

    print(args)

    # explanation mode for nap
    if args.explain:
        preprocessor = Preprocessor(args)

        event_log_size_limit = 100

        #event_log = pd.read_csv(args.data_dir + args.data_set, sep=";", quotechar='|')
        event_log = pd.read_csv(args.data_dir + args.data_set)
        event_log.sort_values(by=[log_params.get("case_id_key", "case"), log_params.get("timestamp_key", 'time')], inplace=True)

        unique_activities = event_log[log_params.get("activity_key", "event")].unique()

        event_log = event_log[:event_log_size_limit]

        #prefix_heatmaps: str = ""
        predictions = []
        temporal_explanations = []
        activity_relevance_explanations = []
        cases = event_log.groupby(log_params.get("case_id_key", "case"))
        for case_id, trace in tqdm(cases, desc="Explain traces"):
            # Skip prediction and explanation for first event
            predictions.append(None)
            temporal_explanations.append([])

            process_instance = [*trace[log_params.get("activity_key", "event")].to_list(), "!"]
            #process_instance = preprocessor.get_random_process_instance(args.rand_lower_bound, args.rand_upper_bound)

            for prefix_index in range(2, len(process_instance)):
                # next activity prediction
                predicted_act_class, predicted_act_class_str, target_act_class, target_act_class_str, prefix_words, model, input_encoded, prob_dist = test.test_prefix(args, preprocessor, process_instance, prefix_index, model_path=args.model_path)
                print("Prefix: %s; Next activity prediction: %s; Next activity target: %s" % (prefix_index, predicted_act_class, target_act_class_str))
                print("Probability Distribution:")
                print(prob_dist)

                # compute lrp relevances
                eps = 0.001  # small positive number
                bias_factor = 0.0  # recommended value
                net = LSTM_bidi(args, model, input_encoded)  # load trained LSTM model

                Rx, Rx_rev, R_rest = net.lrp(prefix_words, predicted_act_class, eps, bias_factor)  # perform LRP
                R_words = np.sum(Rx + Rx_rev, axis=1)  # compute word-level LRP relevances
                scores = net.s.copy()  # classification prediction scores
                print(R_words)
                predictions.append(predicted_act_class_str)
                temporal_explanations.append(list(R_words))

                #prefix_heatmaps = prefix_heatmaps + html_heatmap(prefix_words, R_words) + "<br>"  # create heatmap
                #browser.display_html(prefix_heatmaps)  # display heatmap

        event_log["prediction"] = predictions
        event_log["explanation"] = temporal_explanations
        explanation_path = Path(args.task) / args.result_dir / "local_explanations.csv"
        explanation_path.parent.mkdir(parents=True, exist_ok=True)
        event_log.to_csv(explanation_path, index=False)
        print(f"Saved explanations to {explanation_path.resolve()}")

    # validation mode for nap
    elif not args.explain:

        preprocessor = Preprocessor(args)

        if args.cross_validation:
            for iteration_cross_validation in range(0, args.num_folds):
                preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

                output["training_time_seconds"].append(train.train(args, preprocessor))
                test.test(args, preprocessor)

                output = utils.get_output(args, preprocessor, output)
                utils.print_output(args, output, iteration_cross_validation)
                utils.write_output(args, output, iteration_cross_validation)

            utils.print_output(args, output, iteration_cross_validation + 1)
            utils.write_output(args, output, iteration_cross_validation + 1)

        else:
            output["training_time_seconds"].append(train.train(args, preprocessor))
            test.test(args, preprocessor)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, -1)
            utils.write_output(args, output, -1)

    else:
        print("No mode selected ...")

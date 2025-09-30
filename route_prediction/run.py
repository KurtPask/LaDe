# -*- coding: utf-8 -*-
import json
from pprint import pprint

from utils.util import dict_merge, get_common_params


def run(params):
    pprint(params)
    model = params["model"]

    if model in ["Distance-Greedy", "Time-Greedy", "Or-Tools"]:
        from algorithm.basic.basic_model import main
    elif model == "fdnet":
        from algorithm.fdnet.train import main
    elif model == "deeproute":
        from algorithm.deeproute.train import main
    elif model == "osqure":
        from algorithm.osqure.train import main
    elif model == "graph2route":
        from algorithm.graph2route.train import main
    elif model == "cproute":
        from algorithm.cproute.train import main
    elif model == "m2g4rtp_pickup":
        from algorithm.m2g4rtp_pickup.train import main
    elif model == "drl4route":
        from algorithm.drl4route.train import main
    else:
        raise ValueError(f"Unsupported model: {model}")

    main(params)


def get_params():
    parser = get_common_params()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Name of the model to run. When omitted, run.py executes its default "
            "benchmark sweep."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset identifiers to evaluate. Overrides --dataset.",
    )
    parser.add_argument(
        "--use_tropical_attention",
        action="store_true",
        help="Use the tropical multi-head attention inside CPRoute.",
    )
    parser.add_argument(
        "--tropical_attention_kwargs",
        type=str,
        default=None,
        help=(
            "JSON-encoded dictionary with additional keyword arguments passed to "
            "the tropical attention module."
        ),
    )
    parser.add_argument(
        "--cuda_id",
        type=int,
        default=0,
        help="CUDA device identifier used by compatible algorithms.",
    )
    args, _ = parser.parse_known_args()
    return args


def _parse_tropical_kwargs(params):
    raw_kwargs = params.get("tropical_attention_kwargs")
    if not raw_kwargs:
        params.pop("tropical_attention_kwargs", None)
        return
    if isinstance(raw_kwargs, dict):
        return
    try:
        params["tropical_attention_kwargs"] = json.loads(raw_kwargs)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--tropical_attention_kwargs must be a valid JSON object."
        ) from exc


def _build_default_experiments(base_params, dataset_overrides=None):
    splits = ['jl', 'cq', 'yt', 'sh', 'hz']
    datasets = dataset_overrides or [f"pickup_{i}" for i in splits]
    args_lst = []
    for model in [
        #"Distance-Greedy",
        #"Time-Greedy",
        #"Or-Tools",
        #"osqure",
        #"deeproute",
        #"fdnet",
        #"graph2route",
        "cproute",
        #"m2g4rtp_pickup",
        #"drl4route",
    ]:
        if model in ["Distance-Greedy", "Time-Greedy", "Or-Tools"]:
            for dataset in datasets:
                basic_params = dict_merge([base_params, {"model": model, "dataset": dataset}])
                args_lst.append(basic_params)
        elif model == "osqure":
            for dataset in datasets:
                osqure_params = dict_merge(
                    [base_params, {"model": model, "dataset": dataset}]
                )
                args_lst.append(osqure_params)
        elif model == "drl4route":
            for hs in [64, 32]:
                for rl_r in [0.2, 0.4, 0.6, 0.8, 1]:
                    for dataset in datasets:
                        dl_params = dict_merge(
                            [
                                base_params,
                                {
                                    "model": model,
                                    "hidden_size": hs,
                                    "dataset": dataset,
                                    "rl_ratio": rl_r,
                                },
                            ]
                        )
                        args_lst.append(dl_params)
        elif model in ["deeproute", "fdnet", "cproute", "m2g4rtp_pickup"]:
            for hs in [32, 64]:
                for dataset in datasets:
                    deeproute_params = dict_merge(
                        [
                            base_params,
                            {"model": model, "hidden_size": hs, "dataset": dataset},
                        ]
                    )
                    args_lst.append(deeproute_params)
        elif model == "graph2route":
            for hs in [32, 64]:
                for gcn_num_layers in [2, 3]:
                    for dataset in datasets:
                        for knn in ["n-1", "n"]:
                            graph2route_params = dict_merge(
                                [
                                    base_params,
                                    {
                                        "model": model,
                                        "hidden_size": hs,
                                        "gcn_num_layers": gcn_num_layers,
                                        "worker_emb_dim": 20,
                                        "dataset": dataset,
                                        "k_nearest_neighbors": knn,
                                    },
                                ]
                            )
                            args_lst.append(graph2route_params)
    return args_lst


if __name__ == "__main__":
    params = vars(get_params())
    _parse_tropical_kwargs(params)

    dataset_overrides = params.pop("datasets", None)

    if params.get("model"):
        dataset_names = dataset_overrides or [params.get("dataset")]
        for dataset in dataset_names:
            single_run_params = dict_merge([params, {"dataset": dataset}])
            run(single_run_params)
    else:
        args_lst = _build_default_experiments(params, dataset_overrides)
        for p in args_lst:
            run(p)










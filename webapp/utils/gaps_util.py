"""
    Apache License 2.0 Copyright (c) 2022
    gaps filling utility using Fedot api
"""
import logging
import numpy as np
import pandas as pd
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.utilities.ts_gapfilling import ModelGapFiller


def get_composite_pipeline():
    """ Returns prepared pipeline of 5 models """
    node_1 = PipelineNode("lagged")
    node_1.parameters = {"window_size": 200}
    node_2 = PipelineNode("lagged")
    node_2.parameters = {"window_size": 192}
    node_linear_1 = PipelineNode("linear", nodes_from=[node_1])
    node_linear_2 = PipelineNode("linear", nodes_from=[node_2])

    node_final = PipelineNode("ridge",
                              nodes_from=[node_linear_1, node_linear_2])
    pipeline = Pipeline(node_final)
    return pipeline


def get_simple_pipeline():
    """ Returns simple ridge regression pipeline """
    node_lagged = PipelineNode("lagged")
    node_lagged.parameters = {"window_size": 192}
    node_ridge = PipelineNode("ridge", nodes_from=[node_lagged])
    ridge_pipeline = Pipeline(node_ridge)
    return ridge_pipeline


def fill_gaps(file_path="data/sunspot_gaps.csv"):
    """ Fill gaps in timeseries instead of -1 values. """
    try:
        dataframe = pd.read_csv(file_path, delimiter=";")
        dataframe["date"] = dataframe["date"].astype(float)

        simple_pipe = get_simple_pipeline()
        filler1 = ModelGapFiller(gap_value=-1.0, pipeline=simple_pipe)
        gaps = np.array(dataframe["with_gap"])
        dataframe["ridge"] = filler1.forward_inverse_filling(gaps)

        pipeline = get_composite_pipeline()
        filler2 = ModelGapFiller(gap_value=-1.0, pipeline=pipeline)
        dataframe["composite"] = filler2.forward_filling(gaps)
        return dataframe
    except FileNotFoundError as exc:
        logging.error(f"ошибка: файл не найден {file_path}")
        raise exc

"""
---------------- UniversalPipeline: a flat Composite of processors, each reading columns of one frame ----------
-------------------- purpose is to track variables between raw and encoded data -----------------------
"""

import concurrent.futures
import logging
import multiprocessing
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from polyergalio.encoders.encoders import Processor
from polyergalio.composite_model import Composite, CompositeNode

log = logging.getLogger(__name__)


def convert_datastructure_for_processing(data_object) -> pd.DataFrame:
    """
    Conforms incoming data into the DF format we're expecting
    Parameters
    ----------
    data_object : dict, array, series of df of data

    Returns
    -------
    DataFrame structured from the input data
    """
    if isinstance(data_object, pd.DataFrame):
        return data_object
    elif isinstance(data_object, dict):
        variables = [str(k) for k in data_object.keys()]
        all_values = [vals for vals in data_object.values()]
    elif isinstance(data_object, pd.Series):
        variables = [str(data_object.name)]
        all_values = data_object.values.reshape(1, -1)
    elif isinstance(data_object, np.ndarray):
        variables = [str(i) for i in range(0, data_object.shape[0])]
        all_values = data_object

    variable_count = len(variables)
    all_lengths = [len(v) for v in all_values]

    assert variable_count == len(all_lengths)
    assert np.all(np.equal(all_lengths, all_lengths[0]))

    return pd.DataFrame.from_dict({k: v for k, v in zip(variables, all_values)})


def expand_list_col(results: dict, meta_dict: dict) -> pd.DataFrame:
    """
    Take the encoded results and flatten out any multi-dimensional columns.
    eg for a text column, we'll have an encoded list of ints in one dataframe cell
    Parameters
    ----------
    results: the result data from the encoding process
    meta_dict: the metadata dictionary from the pipeline

    Returns
    -------
    dataframe with multidimensional columns expanded (flattened)
    """
    list_of_dicts = []
    for metadata in meta_dict.values():
        out = metadata["output_dimension"]
        if out == len(metadata["variable_names"]):
            for v_name in metadata["variable_names"]:
                if v_name in results.keys():
                    list_of_dicts.append({v_name: results[v_name]})
                    continue
                else:
                    raise KeyError(f"{v_name} not present in our result data")
        else:
            (v_name,) = metadata["variable_names"]
            num_columns = metadata["output_dimension"]
            _results = np.array(results.pop(v_name))
            for i in range(0, num_columns):
                list_of_dicts.append({f"{v_name}_{i}": _results[:, i]})

    return pd.DataFrame({k: v for one_d in list_of_dicts for k, v in one_d.items()})


def combine_futures(futures) -> dict:
    """
    helper function for multithreading / multiprocessing the encoding process
    Parameters
    ----------
    futures : future processes created with multiprocessing

    Returns
    -------
    dict of the results
    """
    combined = {}
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if isinstance(result, bool):
            continue
        combined.update(result)
    return combined


def output_names(processor: Processor) -> list[str]:
    """
    Names of the columns a processor encodes to; a one-to-many encoder gets a numbered name per column.

    Raises
    ------
    ValueError
        if the processor's metadata does not follow the pipeline contract
    """
    try:
        (entry,) = processor.metadata.values()
        names, width = entry["variable_names"], entry["output_dimension"]
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{processor.__class__.__name__}.metadata does not follow the pipeline contract "
            "({target: {output_dimension, variable_names, enc_type, ...}})"
        ) from error
    if len(names) != width:
        names = [f"{names[0]}_{idx}" for idx in range(width)]
    return list(names)


class ProcessorNode(CompositeNode):
    """A processor and the input columns it reads, in the order its encode takes them."""

    @property
    def processor(self) -> Processor:
        return self.component

    @property
    def outputs(self) -> dict:
        """the processor's metadata entry: output dimension, variable names, encoder type"""
        (entry,) = self.component.metadata.values()
        return entry

    @property
    def output_names(self) -> list[str]:
        """names of the encoded columns; a one-to-many encoder gets a numbered name per column"""
        return output_names(self.component)


class UniversalPipeline(Composite):
    """
    Every processor encoding its own columns of one frame, all at once.

    Connect processors one by one, then fit on a frame and encode. The
    encoded matrix lays out each node's columns in connection order, and
    connections can be added at any point:

        pipeline = UniversalPipeline()
        pipeline.connect(NuemricNormalizeProcessor("age"))
        pipeline.connect(TimeAbsoluteProcessor("end"), "end", "start", "tz", name="duration")
        pipeline.fit(frame)
        encoded = pipeline.encode(frame)
    """

    component_family = Processor

    def connect(
        self, processor: Processor, *columns, name: Optional[str] = None, strict: bool = True
    ) -> ProcessorNode:
        """
        Place a processor in the pipeline, reading the given columns.

        Parameters
        ----------
        processor : the processor to run at this node
        columns : input columns, in the order the processor's encode takes
            them. Defaults to the processor's target.
        name : optional label, unique in this pipeline. Defaults to the
            processor's target.
        strict : False places the processor without requiring its columns,
            as an unwired node that reconnect() can feed later; an unwired
            node is left out of fit, encode and the output layout.

        Returns
        -------
        the new node
        """
        columns = columns or ((processor.target,) if strict else ())
        return super().connect(processor, *columns, name=name, strict=strict)

    def entry_point(self, processor: Processor):
        return processor.encode

    def check_connection(self, processor: Processor, sources: tuple, strict: bool = True, replacing=None) -> None:
        for position, column in enumerate(sources):
            if not isinstance(column, (str, int)):
                raise TypeError(f"column {position} is {type(column).__name__}, expected a column name")
        if not strict:
            return

        taken = {
            name: node.name
            for node in self.active_nodes
            if node is not replacing
            for name in node.output_names
        }
        for name in output_names(processor):
            if name in taken:
                raise ValueError(f"{name!r} is already produced by node {taken[name]!r}; give the processor another target")

        required, maximum = self.positional_counts(processor.encode)
        if len(sources) < required or (maximum is not None and len(sources) > maximum):
            raise ValueError(
                f"{processor.__class__.__name__}.encode takes {required} to {maximum} "
                f"columns, got {len(sources)}"
            )

    def make_node(self, name: str, processor: Processor, sources: tuple) -> ProcessorNode:
        return ProcessorNode(name, processor, sources)

    def default_name(self, processor: Processor, sources: tuple) -> str:
        name = str(processor.target)
        if any(node.name == name for node in self._nodes):
            return self.auto_name(processor)
        return name

    @property
    def processors(self) -> list[Processor]:
        return self.components

    # ------------- data
    def columns_for(self, node: ProcessorNode, frame: pd.DataFrame) -> list[pd.Series]:
        """the input columns a node reads, in connection order"""
        return [frame[column] for column in node.sources]

    def missing_columns(self, frame: pd.DataFrame) -> list:
        """input columns some node reads that the frame does not have"""
        return sorted({column for node in self.active_nodes for column in node.sources if column not in frame.columns}, key=str)

    def fit(self, input_data, unfitted_only: bool = False) -> bool:
        """
        Fit each processor on the columns it reads.

        Parameters
        ----------
        input_data : frame, dict or array holding every column a node reads
        unfitted_only : leave processors that are already fitted alone, e.g.
            to fit nodes connected after an earlier fit

        Returns
        -------
        True once every processor reports fitted
        """
        frame = convert_datastructure_for_processing(input_data)
        missing = self.missing_columns(frame)
        if missing:
            raise KeyError(f"input has no column(s) {missing}")

        for node in self.active_nodes:
            if unfitted_only and node.processor.is_fitted:
                continue
            node.processor.fit(*self.columns_for(node, frame))
            if not node.processor.is_fitted:
                raise ValueError(f"{node.name} did not report fitted after fit")
        return True

    def encode(self, input_data, use_multiprocess: bool = False) -> pd.DataFrame:
        """
        If you run into issues, first try setting multiprocess to False to get more visibility. Multiprocessing
        always makes it more challanging to track down the issue!

        Parameters
        ----------
        input_data : frame, dict or array holding every column a node reads
        use_multiprocess : encode nodes in parallel processes, useful with a very large number of nodes

        Returns
        -------
        DataFrame of encoded values, each node's columns in connection order
        """
        unfitted = [node.name for node in self.active_nodes if not node.processor.is_fitted]
        if unfitted:
            raise ValueError(f"not yet fitted: {unfitted}")

        frame = convert_datastructure_for_processing(input_data)
        missing = self.missing_columns(frame)
        if missing:
            raise KeyError(f"input has no column(s) {missing}")

        results = {}
        if use_multiprocess:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=multiprocessing.cpu_count() // 2,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                tasks = {
                    executor.submit(node.processor.encode, *self.columns_for(node, frame)): node.name
                    for node in self.active_nodes
                }
                results = combine_futures(tasks)
        else:
            for node in self.active_nodes:
                results.update(node.processor.encode(*self.columns_for(node, frame)))

        return expand_list_col(results, self.metadata)

    def decode(self, input_data: dict) -> dict:
        """
        Decode model predictions back into the original representation.

        Parameters
        ----------
        input_data : node name -> encoded values; nodes without an entry are skipped

        Returns
        -------
        node name -> decoded values
        """
        decoded = {}
        for node in self.active_nodes:
            if node.name in input_data:
                decoded[node.name] = node.processor.inverse(np.asarray(input_data[node.name]).astype(np.object_))
        return decoded

    @property
    def is_fitted(self) -> bool:
        unfitted = [node.name for node in self.active_nodes if not node.processor.is_fitted]
        for name in unfitted:
            log.warning(f"{name} has not been fitted")
        return not unfitted

    # ------------- description
    @property
    def metadata(self) -> dict:
        """node name -> that processor's metadata entry"""
        return {node.name: node.outputs for node in self.active_nodes}

    def layout(self) -> list[tuple[ProcessorNode, int, int]]:
        """(node, first column, column count) for every node, in the encoded matrix"""
        rows, start = [], 0
        for node in self.active_nodes:
            width = node.outputs["output_dimension"]
            rows.append((node, start, width))
            start += width
        return rows

    @property
    def output_mapping(self) -> dict:
        """node name -> (start, end) columns of the encoded matrix"""
        return {node.name: (start, start + width) for node, start, width in self.layout()}

    @property
    def output_column_names(self) -> list[str]:
        """names of the encoded columns"""
        return [name for node, _, _ in self.layout() for name in node.output_names]

    @property
    def output_column_types(self) -> list[str]:
        """encoder type of each encoded column"""
        return [node.outputs["enc_type"] for node, _, width in self.layout() for _ in range(width)]

    @property
    def get_profiler_output_info(self) -> list[int]:
        """encoded column indices of nodes whose name mentions "profile" """
        return [start for node, start, _ in self.layout() if "profile" in node.name.lower()]

    @property
    def get_categorical_output_info(self) -> dict[int, int]:
        """encoded column index -> number of categories, for categorical nodes"""
        return {
            start: node.outputs["num_categories"]
            for node, start, _ in self.layout()
            if node.outputs["enc_type"] == "categorical"
        }

    @property
    def get_numeric_output_info(self) -> list[int]:
        """encoded column indices of numeric and chronological nodes"""
        return [
            start + offset
            for node, start, width in self.layout()
            if node.outputs["enc_type"] in ("numeric", "chronological")
            for offset in range(width)
        ]

    @property
    def get_text_output_info(self) -> tuple[list[int], int]:
        """encoded column indices of text nodes, and their combined tokenized length"""
        text = [(start, width) for node, start, width in self.layout() if node.outputs["enc_type"] == "text"]
        return [start + offset for start, width in text for offset in range(width)], sum(width for _, width in text)

    def validate(self, input_data=None) -> list[str]:
        """
        Problems that would stop fit or encode.

        Parameters
        ----------
        input_data : when given, also checks the input has every column a node reads
        """
        problems = []
        for node in self._nodes:
            if not self.is_wired(node):
                problems.append(
                    f"{node.name} reads {len(node.sources)} of {self.required_sources(node.processor)} "
                    "columns, so it is left out"
                )
            elif not node.processor.is_fitted:
                problems.append(f"{node.name} has not been fitted")

        if input_data is not None:
            missing = self.missing_columns(convert_datastructure_for_processing(input_data))
            if missing:
                problems.append(f"input has no column(s) {missing}")
        return problems

    def summary(self) -> str:
        lines = [f"{self.name}: {len(self)} nodes, {len(self.output_column_names)} output columns"]
        width = max((len(node.name) for node in self._nodes), default=4)
        for node, start, count in self.layout():
            state = "fitted" if node.processor.is_fitted else "not fitted"
            lines.append(
                f"  {node.name.ljust(width)}  <- {', '.join(str(column) for column in node.sources):<28} "
                f"columns {start}:{start + count}  {state}"
            )
        for node in self._nodes:
            if not self.is_wired(node):
                lines.append(f"  {node.name.ljust(width)}  <- {', '.join(str(column) for column in node.sources):<28} not connected")
        return "\n".join(lines)

    def __str__(self):
        return str(self.metadata)

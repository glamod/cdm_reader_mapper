"""Common Data Model (CDM) pandas duplicate check."""

from __future__ import annotations

import reportlinkage

def dataframe_apply_check(df, columns):
    """DOCUMENTATION:"""
    indexer = reportlinkage.Index()
    indexer.full()
    return indexer

def duplicate_check(data, columns):
    """DOCUMENTATION."""
    if not isinstance(columns, list):
        columns = [columns]
        
    # index is a list of integer positions to select from data
    def dataframe(df, columns):
        return dataframe_apply_index(
            df,
            columns,
        )

    def parser(data_parser, index, out_rejected=False):
        read_params = [
            "chunksize",
            "names",
            "dtype",
            "parse_dates",
            "date_parser",
            "infer_datetime_format",
        ]
        read_dict = {x: data_parser.orig_options.get(x) for x in read_params}
        in_buffer = StringIO()
        if out_rejected:
            out_buffer = StringIO()

        for df in data_parser:
            o = dataframe(df, columns)
            o[0].to_csv(in_buffer, header=False, index=False, mode="a")
            if out_rejected:
                o[1].to_csv(out_buffer, header=False, index=False, mode="a")

        in_buffer.seek(0)
        output = [pd.read_csv(in_buffer, **read_dict)]
        if out_rejected:
            out_buffer.seek(0)
            output.append(pd.read_csv(out_buffer, **read_dict))
        return output

    if not isinstance(data, pd.io.parsers.TextFileReader):
        output = dataframe(data, columns)
    else:
        output = parser(data, columns)

    if len(output) > 1:
        return output
    else:
        return output[0]        
    for column in columns:
        return
        
    
    
    

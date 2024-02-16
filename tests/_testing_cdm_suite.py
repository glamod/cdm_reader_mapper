from __future__ import annotations

from cdm_reader_mapper import cdm_mapper, mdf_reader


def _testing_suite(
    source=None,
    data_model=None,
    sections=None,
    cdm_name=None,
    cdm_subset=None,
    codes_subset=None,
    suffix="exp",
    mapping=True,
    out_path=None,
    **kwargs,
):
    name_ = source.split("/")[-1].split(".")[0]
    if sections:
        if isinstance(sections, str):
            sections = [sections]
        name_ = name_ + "_" + "_".join(sections)
    read_ = mdf_reader.read(
        source=source,
        data_model=data_model,
        out_path=out_path,
        **kwargs,
    )
    data = read_.data
    attrs = read_.attrs
    mask = read_.mask
    dtypes = read_.dtypes
    parse_datetime = read_.parse_datetime

    # data_ = expected_results[suffix]["data"]
    # mask_ = expected_results[suffix]["mask"]

    # for index in data.columns:
    #  pd.testing.assert_series_equal(data[index], data_[index])
    #  pd.testing.assert_series_equal(mask[index], mask_[index])

    if mapping is False:
        return

    output = cdm_mapper.map_model(
        cdm_name,
        data,
        attrs,
        cdm_subset=cdm_subset,
        # codes_subset=codes_subset,
        log_level="DEBUG",
    )

    cdm_mapper.cdm_to_ascii(output, suffix=suffix)
    output = cdm_mapper.read_tables(".", tb_id=suffix)

    # output_ = expected_results[suffix]["cdm"]

    # for column in output.columns:
    #  pd.testing.assert_series_equal(output[column], output_[column])

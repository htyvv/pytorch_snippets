"""
This is the code that test whether the data we preprocessed is valid 
before putting data into influxDB.
"""
import os
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Iterable
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, Hashable


def isfloat(value: str) -> bool:
    """
    Is it possible to convert this string to float type?

    Args:
        value: A string to see if it can be converted to a float

    Returns:
        True: it is possible to convert to float type
    """
    try:
        float(value)
        return True
    except:
        return False


def isbool(value):
    """
    Is it possible to convert this string to bool type?

    Args:
        value: A string to see if it can be converted to a bool

    Returns:
        False: it is impossible to convert to bool type
    """
    try:
        return value.strip().lower() in ("true", "false")
    except:
        return False


class DataFrameTest:
    """
    test whether the data we preprocessed is valid
    """

    def __init__(self, df: pd.DataFrame, tags: List):
        self.df = df
        self.tags = tags
        self.time = "time"
        self.time_df = None
        self.fields = [col for col in df.columns if col not in tags + [time]]
        self.object_columns = [col for col in self.fields if self.df[col].dtypes == object]
        self.unhashable = None
        self.project_name = os.popen("git rev-parse --show-toplevel").read().split("/")[-1][:-1]

        self.__is_time_format_test_ok = False
        self.__is_duplicate_test_ok = False
        self.__is_column_name_test_ok = False
        # self.__is_key_time_validation_test_ok = False
        self.__is_valid_sequence_length_test_ok = False
        self.__is_including_nan_test_ok = False
        self.__is_time_order_test_ok = False

        self.passed_fields = []

        self.count_df = None
        self.duplicated_key_time = None

    def test(self):
        """
        execute all sub-test
        """
        self.__column_name_test()
        self.__time_column_test()
        self.__hashable_test()
        print(
            "unhashable columns:",
            None if not self.unhashable else list(self.unhashable.keys()),
        )
        self.__time_format_test()
        self.__including_nan_test()
        self.__duplicate_test()
        self.__time_order_test()
        self.__valid_sequence_length_test()
        self.__column_data_type_test()

    def __column_name_test(self):
        """Remove spaces and convert to lowercase"""
        print("column_name_test: start", end="")
        if not self.__is_column_name_test_ok:
            self.df.columns = [str(col).strip().lower().replace(" ", "") for col in self.df.columns]
            self.tags = [str(t).strip().lower().replace(" ", "") for t in self.tags]
            self.__is_column_name_test_ok = True
            print("\rcolumn_name_test: pass ")
        else:
            print("\rcolumn_name_test: pass ")

    def __time_column_test(self):
        """
        Check if there is 'time' in dataframe columns
        """
        if "time" not in self.df.columns:
            raise Exception("The 'time' is not within the dataframe columns.")

    def __hashable_test(self):
        """
        find unhashable columns
        """
        self.unhashable = {}
        for col in self.object_columns:
            hashable_series = self.df[col].apply(lambda x: isinstance(x, Hashable))
            hashable_percent = np.mean(hashable_series)
            if hashable_percent < 1:
                self.unhashable[col] = hashable_series

    def __time_format_test(self):
        """
        Check if the type of 'time' column is datetime
        """
        print("time_format_test: start", end="")
        if not self.__is_time_format_test_ok:
            if self.df[self.time].dtypes == "datetime64[ns]":
                print("\rtime_format_test: pass ")
                self.__is_time_format_test_ok = True
            else:
                try:
                    if self.df[self.time].dtypes == "datetime64[ns, UTC]":
                        print("\rtime_format_test: pass ")
                        self.__is_time_format_test_ok = True
                except TypeError:
                    print("\rtime_format_test: fail ")
                    raise Exception("Change the time column to datetime format")
        else:
            print("\rtime_format_test: pass ")

    def __including_nan_test(self):
        """
        Find columns including nan values
        """
        print("including_nan_test: start", end="")
        if not self.__is_including_nan_test_ok:
            nan_columns = list(self.df.columns[np.sum(pd.isna(self.df)) != 0])
            if len(nan_columns) == 0:
                print("\rincluding_nan_test: pass ")
                self.__is_including_nan_test_ok = True
            else:
                print("\rincluding_nan_test: fail ")
                raise Exception(f"A list of columns containing nan values\n\t{nan_columns}")
        else:
            print("\rincluding_nan_test: pass ")

    def __duplicate_test(self):
        """
        If there is a row where all values are the same, remove it
        """
        print("duplicate_test: start", end="")
        if not self.__is_duplicate_test_ok:
            shape = self.df.shape[0]
            self.df.drop_duplicates(
                [col for col in self.df.columns if col not in list(self.unhashable.keys())],
                inplace=True,
            )
            self.df.reset_index(drop=True, inplace=True)
            self.__is_duplicate_test_ok = True
            print(f"\rduplicate_test: pass, row {shape} -> {self.df.shape[0]}")
        else:
            print("\rduplicate_test: pass ")

    def __time_order_test(self):
        """"""
        order_pass = []
        coef_of_var = []
        diff_list = []
        keys = []

        print("time_order_test: start", end="")

        if self.time_df is None:
            for key, temp in tqdm(self.df.groupby(self.tags)):
                keys.append(key)
                sorted_temp = temp.loc[temp.time.sort_values().index]
                diff = np.round(np.diff(temp.time) / pd.Timedelta(seconds=1), 3)
                diff_list.append(diff)

                if isinstance(key, Iterable):
                    key = [str(k) for k in key]
                    name = "_".join(key)
                else:
                    name = str(key)

                if np.mean(temp.index == sorted_temp.index) == 1.0:
                    order_pass.append(True)
                else:
                    order_pass.append(False)
                    self.__draw_subplot(name, temp)

                coef_of_var.append(np.std(diff) / np.mean(diff))

            time_df = pd.DataFrame(data=keys, columns=self.tags)
            time_df["order_pass"] = order_pass
            time_df["coef_of_var"] = coef_of_var
            time_df["diff"] = diff_list
            self.time_df = time_df.sort_values("coef_of_var").reset_index(drop=True)
        else:
            for idx in tqdm(self.time_df[~self.time_df["order_pass"]].index):
                temp = self.df[
                    np.mean(self.df[self.tags] == dict(self.time_df[self.tags].loc[idx]), 1) == 1
                ]
                sorted_temp = temp.loc[temp.time.sort_values().index]
                diff = np.round(np.diff(temp.time) / pd.Timedelta(seconds=1), 3)

                update_dict = dict(self.time_df.loc[idx][self.tags])
                update_dict["order_pass"] = np.mean(temp.index == sorted_temp.index) == 1.0
                update_dict["diff"] = diff
                update_dict["coef_of_var"] = np.std(diff) / np.mean(diff)
                self.time_df.loc[idx] = update_dict

        if not self.__is_time_order_test_ok:
            if np.mean(self.time_df["order_pass"]) == 1:
                print("\rtime_order_test: pass ")
                self.__is_time_order_test_ok = True
            else:
                name = [
                    "/".join(tag_values.astype(str))
                    for tag_values in self.time_df[self.tags].values
                ]
                fig = px.bar(
                    self.time_df,
                    x=np.arange(len(self.time_df)),
                    y="coef_of_var",
                    color=self.time_df["order_pass"],
                    hover_name=name,
                    title="tags: " + ", ".join(self.tags),
                )
                fig.add_hline(y=0.1)
                fig.add_annotation(x=0, y=0.1, text="stable line")
                fig.show()
                print("\rtime_order_test: fail ")
                raise Exception("There are samples out of time order. (see self.time_df)")

    def __valid_sequence_length_test(self):
        """
        Test for selecting samples with valid length.
        """
        print("valid_sequence_length_test: start", end="")

        if not self.__is_valid_sequence_length_test_ok:
            self.count_df = self.df[self.tags + [self.time]].groupby(self.tags).count()
            dist = self.count_df.reset_index().rename(columns={"time": "length"})

            fig = px.histogram(dist, x="length")

            fig.update_layout(xaxis_title_text="sequence length", yaxis_title_text="count")
            fig.show()

            while True:
                self.min_length = int(input("write min length: "))
                self.max_length = int(input("write max length: "))

                dist["label"] = "selected"
                dist["label"][
                    (dist["length"] < self.min_length) | (dist["length"] > self.max_length)
                ] = "discarded"

                fig = px.histogram(dist, x="length", color="label")
                fig.update_layout(xaxis_title_text="sequence length", yaxis_title_text="count")
                fig.show()
                ok = input("Do you want to confirm the length like this? (Y/n)").lower()
                if ok in ["", "y", "yes"]:
                    valid_tags = dist[dist.label == "selected"][self.tags]
                    condition = self.df.apply(
                        lambda x: len(
                            np.where(np.mean(valid_tags.values == x[self.tags].values, 1) == 1)[0]
                        )
                        > 0,
                        axis=1,
                    )
                    shape = self.df.shape[0]
                    self.df = self.df[condition]
                    print(f"\rvalid_sequence_length_test: pass, row {shape} -> {self.df.shape[0]}")

                    self.__is_valid_sequence_length_test_ok = True
                    print("\rvalid_sequence_length_test: pass ")
                    break

        else:
            print("\rvalid_sequence_length_test: pass ")

    def __column_data_type_test(self):
        """
        Check if there is a column of object type with mixed data types
        """
        print("column_data_type_test: start")
        for col in tqdm(self.object_columns):
            if col not in self.passed_fields:
                type_series = self.df[col].apply(lambda x: type(x).__name__)
                count_type = Counter(type_series)

                count_float = np.sum(self.df[col][type_series == "str"].apply(lambda x: isfloat(x)))
                count_bool = np.sum(self.df[col][type_series == "str"].apply(lambda x: isbool(x)))

                if count_float + count_bool != 0:
                    count_type["str"] -= count_float + count_bool
                    count_type["is_float"] = count_float
                    count_type["is_bool"] = count_bool

                if len(count_type.keys()) != 1:
                    print("\t", col, ":", count_type)
                else:
                    self.passed_fields.append(col)

        if self.object_columns == self.passed_fields:
            print("column_data_type_test: pass ")
        else:
            print("column_data_type_test: fail ")

    def __draw_subplot(self, name, temp):
        temp = temp.reset_index(drop=True)
        sorted_temp = temp.sort_values(self.time)
        fig_list = []

        ordered_col = temp.columns[temp.dtypes != object]
        ordered_col = [col for col in ordered_col if len(temp[col].unique()) > 3]

        file_path = "quality_test/time_order"
        os.makedirs(file_path, exist_ok=True)

        for row, col in enumerate(ordered_col):
            sample = temp[col].values
            sorted_sample = sorted_temp[col].values

            fig = make_subplots(subplot_titles=[col], specs=[[{"secondary_y": True}]])

            sample_diff = np.diff(sample)
            sorted_sample_diff = np.diff(sorted_sample)

            line_df = pd.DataFrame(
                {
                    "original": sample,
                    "sorted_line": sorted_sample,
                    "sub": np.hstack(
                        (
                            sample_diff - sorted_sample_diff,
                            np.array([0]).astype(sample_diff.dtype),
                        )
                    ),
                }
            )
            diff = np.round(np.diff(temp.time) / pd.Timedelta(seconds=1), 3)
            where = np.where(diff < 0)[0]

            fig.add_trace(
                go.Scatter(
                    x=line_df.index,
                    y=line_df["original"],
                    name="original",
                    line={"color": "#148CFF"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=line_df.index,
                    y=line_df["sorted_line"],
                    name="sorted_line",
                    line={"dash": "dot"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=line_df.index, y=line_df["sub"], name="sub", line={"color": "#FFC300"}
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(x=where, y=sample[where], mode="markers", name="changed_point")
            )

            fig.update_yaxes(title_text="value", secondary_y=False)
            fig.update_yaxes(title_text="subtract diff", secondary_y=True)
            fig.update_layout(width=1200, height=400, margin=dict(l=400, r=200, t=50, b=50))

            fig_list.append(fig.to_html(include_plotlyjs="cdn", full_html=False))

        name = name.replace("/", "_").replace(":", "_")
        with open(os.path.join(file_path, "%s.html" % name), "w") as f:
            f.write(" ".join(fig_list))

    def result(self):
        """
        Return the test results so far
        """
        print(
            f"""
unhashable columns: {None if not self.unhashable else list(self.unhashable.keys())},
time_format_test: {'pass' if self.__is_time_format_test_ok else 'fail'}, 
including_nan_test: {'pass' if self.__is_including_nan_test_ok else 'fail'},
column_name_test: {'pass' if self.__is_column_name_test_ok else 'fail'}, 
duplicate_test: {'pass' if self.__is_duplicate_test_ok else 'fail'},
valid_sequence_length_test: {'pass' if self.__is_valid_sequence_length_test_ok else 'fail'},
column_data_type_test: {'pass' if self.object_columns == self.passed_fields else 'fail'}"""
        )

    def show_diff_hist(self, num):
        fig = px.histogram(pd.DataFrame({"diff": self.time_df.iloc[num]["diff"]}), x="diff")
        fig.show()

    def change_time_order(self, option="force", select="all"):
        if self.time_df is None:
            self.time_order_test()
        if select == "all":
            select = self.time_df[~self.time_df["order_pass"]][self.tags].values
        else:
            select = self.time_df.iloc[select][self.tags].values

        if option == "force":
            for tag_value in tqdm(select):
                sample = self.__select_sample(tag_value)[self.time]

                start = min(sample).timestamp()
                end = max(sample).timestamp()

                self.df["time"].loc[sample.index] = (
                    pd.DataFrame(np.linspace(start, end, len(sample)) * 1e9)
                    .astype("datetime64[ns]")
                    .values.reshape(-1)
                )
        elif option == "sort":
            sample_list = [
                self.__select_sample(tag_value).sort_values("time") for tag_value in tqdm(select)
            ]
            self.df = pd.concat(sample_list, 0).reset_index(drop=True)
        else:
            raise NotImplementedError

    def __select_sample(self, tag_value):
        sample = self.df[
            np.mean(
                self.df[self.tags] == {tag: t_v for tag, t_v in zip(self.tags, tag_value)},
                1,
            )
            == 1
        ]

        return sample


if __name__ == "__main__":
    """
    example
    """

    temp = pd.DataFrame(
        {
            "a": [2020, 2021, 2022, 2023, 2024],
            "b ": [1, 2, 2.2, 3, 4],
            "c": ["a", "b", "d", "1", "b"],
            "d": ["asd", "true", "1.2", [1, 2], 1],
            "e": [1, 2, 3, 4, 5],
        }
    )

    print(temp)
    temp["a"] = pd.to_datetime(temp["a"], format="%Y")
    test = DataFrameTest(temp, ["e"], "a")
    print("-----------------------------------------------------")
    test.test()

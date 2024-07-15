"""
Module for feature engineering.
"""

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.job_language_universe = self.config.job_language_universe
        self.job_role_universe = self.config.job_role_universe
        self.relevant_features = self.config.features

    def engineer_features(self, extracted_features):
        talent_data = extracted_features["talent_info"]
        job_data = extracted_features["job_info"]
        language_features = self._engineer_language_features(talent_data["pdf_languages"], job_data["pdf_languages"])
        maturity_features = self._engineer_maturity_features(talent_data["maturity"], job_data["maturity"])
        role_features = self._engineer_role_features(talent_data["pdf_roles"], job_data["pdf_roles"])
        features = pd.concat([language_features, maturity_features, role_features], axis=1)
        features = features[self.relevant_features]
        return features

    def _engineer_role_features(self, pdf_talent_data: pd.DataFrame, pdf_job_data: pd.DataFrame) -> pd.DataFrame:
        """
        This method is responsible for engineering role-related features from the provided talent and job data.

        Args:
            pdf_talent_data: A DataFrame containing talent data with columns 'job_role' and 'indicator'.
            pdf_job_data: A DataFrame containing job data with columns 'job_role' and 'indicator'.

        Returns:
            pd.DataFrame: A DataFrame containing engineered role-related features.

        Notes:
            - It has been decided to use for the p_role_match feature the job role requirements as denominator.
              It would also be possible to use the talent role requirements here, but the modeling results were considered
              to be good enough using the current approach.
        """
        pdf_talent_data["index"] = 1
        pdf_job_data["index"] = 1

        pdf_talent_job_roles_wide = (
            pdf_talent_data.pivot(index="index", columns="job_role", values="indicator").fillna(0).reset_index()
        )
        pdf_talent_job_roles_wide, rel_cols_talent = self._reformat_role_features(pdf_talent_job_roles_wide, "TALENT")

        pdf_job_job_roles_wide = (
            pdf_job_data.pivot(index="index", columns="job_role", values="indicator").fillna(0).reset_index()
        )

        pdf_job_job_roles_wide, rel_cols_job = self._reformat_role_features(pdf_job_job_roles_wide, "JOB")

        pdf_job_roles = pd.concat([pdf_job_job_roles_wide, pdf_talent_job_roles_wide], axis=1)

        pdf_role_match = pd.DataFrame(
            pdf_job_roles[rel_cols_talent].to_numpy() * pdf_job_roles[rel_cols_job].to_numpy(),
            columns=self.job_role_universe,
        )

        pdf_job_roles["p_role_match"] = (
            pdf_role_match.sum(axis=1).iloc[0] / pdf_job_roles[rel_cols_job].sum(axis=1).iloc[0]
        )

        return pdf_job_roles

    def _reformat_role_features(self, pdf: pd.DataFrame, identifier: str) -> (pd.DataFrame, list):
        """
        This method reformats the role features DataFrame by filling missing columns, selecting relevant columns,
        and renaming the columns by adding the identifier as suffix.

        Args:
            pdf: The DataFrame containing role features.
            identifier: The identifier to be appended to the column names.

        Returns:
            Tuple
                pd.DataFrame: The reformatted DataFrame with missing columns filled, relevant columns selected,
                             and renamed columns.
                list: A list of the renamed column names.

        Note:
        - This method uses the _fill_frame method to fill missing columns.
        - The relevant columns are selected based on the job_role_universe attribute.
        - The renamed columns are formed by appending the identifier to the relevant job roles.
        """
        pdf = self._fill_frame(pdf, self.job_role_universe)
        pdf = pdf[self.job_role_universe]
        rel_cols = [f"{rel_job}_{identifier}" for rel_job in self.job_role_universe]
        pdf.columns = rel_cols
        return pdf, rel_cols

    def _engineer_language_features(
        self, pdf_talent_languages: pd.DataFrame, pdf_job_languages: pd.DataFrame
    ) -> pd.DataFrame:
        pdf_talent_languages_wide = self._engineer_talent_language_features(pdf_talent_languages)
        pdf_job_languages_wide = self._engineer_job_language_features(pdf_job_languages)
        pdf_languages_wide = pd.concat([pdf_talent_languages_wide, pdf_job_languages_wide], axis=1)
        pdf_languages_wide["German_must_have_discrepancy"] = (
            pdf_languages_wide["German_must_have_JOB"] - pdf_languages_wide["German_TALENT"]
        )
        pdf_languages_wide["English_must_have_discrepancy"] = (
            pdf_languages_wide["English_must_have_JOB"] - pdf_languages_wide["English_TALENT"]
        )
        pdf_languages_wide["German_should_have_discrepancy"] = (
            pdf_languages_wide["German_should_have_JOB"] - pdf_languages_wide["German_TALENT"]
        )
        pdf_languages_wide["English_should_have_discrepancy"] = (
            pdf_languages_wide["English_should_have_JOB"] - pdf_languages_wide["English_TALENT"]
        )
        return pdf_languages_wide

    def _engineer_talent_language_features(self, pdf_talent_languages: pd.DataFrame) -> pd.DataFrame:
        pdf_talent_languages_required = pdf_talent_languages[
            pdf_talent_languages["title"].isin(self.job_language_universe)
        ]

        pdf_talent_languages_required["index"] = 1
        pdf_talent_languages_required_wide = (
            pdf_talent_languages_required.pivot(index="index", columns="title", values="rating_rank")
            .fillna(0)
            .reset_index()
        )

        pdf_talent_languages_required_wide = self._fill_frame(
            pdf_talent_languages_required_wide, self.job_language_universe
        )

        pdf_talent_languages_required_wide = pdf_talent_languages_required_wide.rename(
            columns={"English": "English_TALENT", "German": "German_TALENT"}
        )

        return pdf_talent_languages_required_wide

    def _engineer_job_language_features(self, pdf_job_languages: pd.DataFrame) -> pd.DataFrame:
        pdf_job_languages["index"] = 1
        pdf_job_languages_must_have_wide = (
            pdf_job_languages[pdf_job_languages["must_have"] == True]
            .pivot(index="index", columns="title", values="rating_rank")
            .fillna(0)
            .reset_index()
        )

        pdf_job_languages_must_have_wide = self._fill_frame(
            pdf_job_languages_must_have_wide, self.job_language_universe
        )

        pdf_job_languages_must_have_wide = pdf_job_languages_must_have_wide.rename(
            columns={"English": "English_must_have_JOB", "German": "German_must_have_JOB"}
        )

        pdf_job_languages_should_have_wide = (
            pdf_job_languages[pdf_job_languages["must_have"] == False]
            .pivot(index="index", columns="title", values="rating_rank")
            .fillna(0)
            .reset_index()
        )

        pdf_job_languages_should_have_wide = self._fill_frame(
            pdf_job_languages_should_have_wide, self.job_language_universe
        ).reset_index()

        pdf_job_languages_should_have_wide = pdf_job_languages_should_have_wide.rename(
            columns={"English": "English_should_have_JOB", "German": "German_should_have_JOB"}
        )

        return pd.concat([pdf_job_languages_must_have_wide, pdf_job_languages_should_have_wide], axis=1)

    @staticmethod
    def _engineer_maturity_features(talent_maturity: dict, job_maturity: dict) -> pd.DataFrame:
        pdf_talent_maturity = pd.DataFrame(talent_maturity, index=[0])
        pdf_job_maturity = pd.DataFrame(job_maturity, index=[0])
        pdf_maturity = pd.concat([pdf_talent_maturity, pdf_job_maturity], axis=1)
        pdf_maturity["salary_discrepancy"] = pdf_maturity["salary_TALENT"] - pdf_maturity["salary_JOB"]
        pdf_maturity["degree_discrepancy"] = pdf_maturity["degree_rank_TALENT"] - pdf_maturity["degree_rank_JOB"]
        pdf_maturity["seniority_rank_discrepancy"] = (
            pdf_maturity["seniority_rank_TALENT"] - pdf_maturity["seniority_rank_JOB"]
        )
        return pdf_maturity

    @staticmethod
    def _fill_frame(pdf: pd.DataFrame, target_cols: list) -> pd.DataFrame:
        """
        This method extends the input pdf with non-existing columns listed in target-cols.
        The new columns are filled with zeros.

        Args:
            pdf: The DataFrame to fill missing columns.
            target_cols: A list of column names that should be present in the DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing columns filled with zeros.

        Note:
        - If the input DataFrame is empty, a new DataFrame with zeros is created.
        - The ordering of the columns in the target_cols list is not guaranteed to be preserved.
        """
        cols = list(pdf)
        missing_cols = list(set(target_cols) - set(cols))
        if len(missing_cols) > 0:
            if len(pdf) == 0:
                pdf = pd.DataFrame(np.zeros((1, len(missing_cols))), columns=missing_cols, index=[0])
            else:
                pdf[missing_cols] = np.zeros(len(missing_cols))

        return pdf

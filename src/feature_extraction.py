"""
Module for feature extraction
"""

import pandas as pd

pd.options.mode.chained_assignment = None


class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.seniority_rank_mapping = self.config.seniority_rank_mapping
        self.pdf_seniority_rank_mapping = self._create_seniority_rank_mapping()
        self.degree_rank_mapping = self.config.degree_rank_mapping
        self.job_role_universe = self.config.job_role_universe
        self.language_rating_rank_mapping = self.config.language_rating_rank_mapping
        self.pdf_language_rating_rank_mapping = self._create_language_rank_mapping()

    def _create_seniority_rank_mapping(self):
        return pd.DataFrame(list(self.seniority_rank_mapping.items()), columns=["seniority", "seniority_rank"])

    def _create_language_rank_mapping(self):
        return pd.DataFrame(list(self.language_rating_rank_mapping.items()), columns=["rating", "rating_rank"])

    def extract_features(self, talent_data, job_data):
        talent_info = self._extract_talent_info(talent_data)
        job_info = self._extract_job_info(job_data)

        return {"talent_info": talent_info, "job_info": job_info}

    def _extract_talent_info(self, talent_data: dict) -> dict:
        pdf_roles = self._extract_roles(talent_data)
        pdf_languages = self._extract_languages(talent_data)
        degree_rank = self._extract_degree(talent_data, "degree")
        seniority_rank = self._extract_seniority(talent_data, multiple=False)
        salary = self._extract_salary(talent_data, key="salary_expectation")

        return {
            "pdf_roles": pdf_roles,
            "pdf_languages": pdf_languages,
            "maturity": {
                "degree_rank_TALENT": degree_rank,
                "seniority_rank_TALENT": seniority_rank,
                "salary_TALENT": salary,
            },
        }

    def _extract_job_info(self, job_data: dict) -> dict:
        pdf_roles = self._extract_roles(job_data)
        pdf_languages = self._extract_languages(job_data)
        degree_rank = self._extract_degree(job_data, "min_degree")
        seniority_rank = self._extract_seniority(job_data, multiple=True)
        salary = self._extract_salary(job_data, key="max_salary")

        return {
            "pdf_roles": pdf_roles,
            "pdf_languages": pdf_languages,
            "maturity": {"degree_rank_JOB": degree_rank, "seniority_rank_JOB": seniority_rank, "salary_JOB": salary},
        }

    def _extract_languages(self, data: dict) -> pd.DataFrame:
        languages = data["languages"]
        pdf_languages = pd.DataFrame(languages)
        pdf_languages = pdf_languages.merge(self.pdf_language_rating_rank_mapping, on="rating", how="inner")
        return pdf_languages

    def _extract_roles(self, data):
        job_roles = data["job_roles"]
        pdf_job_roles = pd.DataFrame(job_roles, columns=["job_role"])
        pdf_job_roles = pdf_job_roles[pdf_job_roles["job_role"].isin(self.job_role_universe)]
        pdf_job_roles["indicator"] = 1
        return pdf_job_roles

    def _extract_degree(self, data, key) -> float:
        return self.degree_rank_mapping.get(data[key])

    def _extract_seniority(self, data: dict, multiple: bool = True) -> float:
        if multiple:
            pdf_seniorities = pd.DataFrame(data["seniorities"], columns=["seniority"])
            pdf_job_seniorities = pdf_seniorities.merge(self.pdf_seniority_rank_mapping, on="seniority", how="inner")
            seniority_rank = pdf_job_seniorities["seniority_rank"].mean()
        else:
            seniority_rank = self.seniority_rank_mapping.get(data["seniority"])

        return seniority_rank

    @staticmethod
    def _extract_salary(data, key):
        return data[key]

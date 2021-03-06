from readability import Readability
from typing import Dict, List, Any


class TextStat:
    """
    The :class:`TextStat` assits users in creating the readability indices using the text data.
    This information in this case is mostly the aggregated text value, which can serve as a multi-author
    large text. Please note that these readability indices were provided, for example, mainly to assess the
    readability of large documents such as navy manuals. They are, therefore, appropriate for large multi-author
    documents.
    """
    def __init__(self):
        """
        Constructor
        """
        self.meta = {
            'coleman_liau': ['grade_level', 'score'],
            'flesch_kincaid': ['grade_level', 'score'],
            'dale_chall': ['grade_levels', 'score'],
            'gunning_fog': ['grade_level', 'score'],
            'smog': ['grade_level', 'score'],
            'spache': ['grade_level', 'score'],
            'linsear_write': ['grade_level', 'score'],
            'ari': ['grade_levels', 'score', 'ages']
        }

    def analyze_text_complexity(self, text: str) -> Dict[str, List[Any]]:
        """
        Given a string representing the text of a document to be considered, the :func:`analyze_text_complexity`
        computes different readability-related evaulations.

        Parameters
        -----------
        text: `str`, required
            The input text to be processed

        Returns
        -----------
        The different grade_level and scores associated with each metric will be returned.
        """
        output = dict()
        r = Readability(text)
        for met in self.meta.keys():
            try:
                r_obj = getattr(r, met)()
            except:
                r_obj = None
            for attr in self.meta[met]:
                key = "_".join([met, attr])
                if r_obj:
                    output[key] = [getattr(r_obj, attr)]
                else:
                    output[key] = ['NA']

        return output

    def analyze_text_by_single_method(self, text: str, method: str, attribute: str) -> Any:
        """
        Given a string representing the text of a document to be considered, the :func:`analyze_text_by_single_method`
        will process the text using one specific chosen method

        Parameters
        -----------
        text: `str`, required
            The input text to be processed

        method: `str`, required
            Method name

        attribute: `str`, required
            The attribute name (for example, score)

        Returns
        -----------
        The different grade_level and scores associated with each metric will be returned.
        """
        r = Readability(text)
        r_obj = getattr(r, method)()
        if r_obj is None:
            return "NA"
        else:
            return getattr(r_obj, attribute)
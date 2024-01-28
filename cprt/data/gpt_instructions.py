SUMMARY_INSTRUCTIONS = """You will receive details about a specific protein. Perform the following tasks:
1) Provide a factual summary, with a maximum of 500 words, that accurately and scientifically describes the functional,
biochemical and structural properties of this protein based only on the provided information.
Ensure that the summary follows a natural and scientific flow, starting with general information such as structure,
localization and taxonomy before detailing functional and biochemical properties.
Ensure that all key points are covered and DON'T provide any further information than what is stated in the input.
2) For each type of information provided, create a question-and-answer pair to elucidate an aspect of
the protein's functionality or biochemical properties without using the protein's name.

Use the below ; separated two-element tuples structure for the output:
(summary: this protein...); (what is question 1?, the answer 1); (what is question 2?, the answer 2); ...

For both tasks if the input contains large group of properties only provide the canonical and crucial information
rather than enumerating every single entry.
DON'T use any of your knowledge to add additional context or information. DON'T add any speculative or unwarranted
information that is not specifically provided.
AVOID using generic phrases or embellished terms such as 'highly', 'several', 'diverse' and 'various'.
"""

METRIC_INSTRUCTIONS = """You will receive details about a specific protein.
Provide a single word answer to the following questions. Print each question and your answer in the same new line.
If the question does not apply to the protein, ignore the question.
1) Is this protein localized to the cell membrane?
2) Is this a membrane protein?
3) Is this protein localized to nucleus?
4) Is this protein localized to mitochondria?
5) Does this protein bind to DNA?
6) Does this protein bind to RNA?
7) Is this protein an enzyme?
8) What are co-factors of this protein as a comma separated list?
"""

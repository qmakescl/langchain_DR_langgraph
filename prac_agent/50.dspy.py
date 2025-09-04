from dotenv import load_dotenv
load_dotenv()

import os
import dspy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key=os.environ["GOOGLE_API_KEY"])
dspy.configure(lm=lm)

math = dspy.ChainOfThought("question -> answer: float")
math(question="Two dice are tossed. What is the probability that the sum equals two?")
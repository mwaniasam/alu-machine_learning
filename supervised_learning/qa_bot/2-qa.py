#!/usr/bin/env python3
"""
Answer Questions
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text
    """
    while True:
        prompt = input("Q: ")
        if prompt.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        answer = question_answer(prompt, reference)
        if answer is None or answer == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))

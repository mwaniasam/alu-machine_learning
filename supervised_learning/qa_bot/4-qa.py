#!/usr/bin/env python3
"""
Multi-reference Question Answering
"""
qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts
    """
    while True:
        prompt = input("Q: ")
        if prompt.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, prompt)
        answer = qa(prompt, reference)

        if answer is None or answer == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))

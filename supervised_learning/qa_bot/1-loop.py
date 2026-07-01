#!/usr/bin/env python3
"""
Create the loop
"""


if __name__ == '__main__':
    while True:
        prompt = input("Q: ")
        if prompt.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break
        print("A:")

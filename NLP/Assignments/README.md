# Workflow to complete an assignment
### step 1
Add a `.md` file and write assignment objective in it (eg. `assignment1.md`)

### step 2
Use Github copilot to generate the `py` file for the assignment and write all the related notes in the `.md` file already present.

#### Prompts
```
Hey, I have added an `assignment1.md` file in which I have written assignment objective. Please read the objective properly and write relevant code for the assignment in `assignment1.py` file and also generate notes in the `assignment1.md` file, you can create multiple `.py` files if needed, but try to do it in one file only, create multiple `.py` files only if there are multiple concepts asked or the code is getting to big that makes it hard to understand, don't write the code too production ready and long, I need a code that can easily explain the concept.

Please remember that the notes in the `assignment1.md` file should not be written in reference with `assignment1.py` file, just write general notes which contains all the necessary steps involved and some important code snippet, the notes should be written in a way that if someone reads it, then he/she would be able to implement the same concepts again (it should be like a blog in markdown format).

and one more thing, clear everything in the `assignment1.md` file before writting the blog/notes in it (like remove all the headings and objective, etc., it was only for the purpose of your information)
```

### step 3
Copy and paste the genearted `md` notes in the gemini chat to create a doc for the assignment.

[Gemini chat Link](https://gemini.google.com/app/97f310497d2de2f1?hl=en-IN)
[Gemini chat Link](https://gemini.google.com/u/1/app/34b16650ea1872cd?hl=en-IN&pageId=none&pli=1)

# Workflow 2
- Paste the prompt with the assignment objective in claude
- Download the `py` and `md` file from claude
- Copy and paste the `md` file in the gemini and tell him to generate the doc
```
# Prompt for gemini

Hey, can you please create a docx file for me of this md notes
```
import pandas as pd
import openai
import numpy as np
import pickle
from flask_cors import CORS, cross_origin

openai.api_key = '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'

COMPLETIONS_MODEL = "text-davinci-003"




MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
separator_len = 3


def get_embedding(text: str, model: str):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str):
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str):
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame):
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }


def vector_similarity(x, y):
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt_follow(question: str, previous_questions, previous_chosen_sections, previous_sources, context_embeddings, df: pd.DataFrame):
    """
    Fetch relevant 
    """
    all_questions = previous_questions
    all_questions.append(question)

    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = previous_chosen_sections
    chosen_sections_len = 0
    chosen_sections_indexes = []

    sources = previous_sources

    counter = 0
    for confidence, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
        title, url = section_index
        source = confidence, title, url
        sources.append(source)
        counter += 1
        if counter >= 2:
            break
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n " + '\n'.join(all_questions) + "\n A:", sources, chosen_sections, all_questions


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}


def format_answer(response, sources):
    sources = [f'{confidence*100:.2f}: {title}: {url}' for confidence, title, url in sources]

    simple_answer = response["choices"][0]["text"].strip(" \n")

    return simple_answer.lstrip() + '\n\nMy answer is based on these sources:\n- ' + '\n- '.join(sources), simple_answer



class GPTBot:
    def __init__(self):
        self.all_questions = []
        self.chosen_sections = []
        self.sources = []
        self.df = pd.read_csv('all_biz.csv')
        self.df = self.df.set_index(["title", 'url'])
        print(f'The dataset has {len(self.df)} entries')
        # self.document_embeddings = compute_doc_embeddings(self.df)
        # with open('embeddings.pickle', 'wb') as handle:
        #     pickle.dump(self.document_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('embeddings.pickle', 'rb') as handle:
            self.document_embeddings = pickle.load(handle)
        print('Done calculating embeddings.')

    def respond_to(self, user_question):
        question = 'Q: ' + user_question
        
        prompt, self.sources, self.chosen_sections, self.all_questions = construct_prompt_follow(
            question = question,
            previous_questions = self.all_questions,
            previous_chosen_sections = self.chosen_sections,
            previous_sources = self.sources,
            context_embeddings = self.document_embeddings,
            df = self.df
        )
        
        response = openai.Completion.create(
                    prompt=prompt,
                    **COMPLETIONS_API_PARAMS
                )
        answer, simple_answer = format_answer(response, self.sources)
        self.all_questions.append('A: ' + simple_answer)
        return answer



from flask import Flask, request

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

bots = {}

@app.route('/api', methods=['POST'])
@cross_origin()
def handle_json():
    json_data = request.get_json()
    session_id = json_data['session']
    question = json_data['question']
    if session_id not in bots:
        bots[session_id] = GPTBot()
    bot = bots[session_id]
    response = bot.respond_to(question)
    return response



if __name__ == '__main__': app.run(debug=True)
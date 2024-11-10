from openai import OpenAI
import google.generativeai as genai
import anthropic
from transformers import pipeline


class llm_summarizer:
    def __init__(self, openai_key, anthropic_key, gemini_key):
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key
        self.gemini_key = gemini_key
        genai.configure(api_key = gemini_key)
        self.openai = OpenAI(api_key = openai_key)
        self.claude = anthropic.Anthropic(api_key = anthropic_key)

    def summarize_bart(self, article_text, summaries, device):
        """
        Summarize article using Bart pretrained transformer

        Args:
            article_text (str): The full text of the article to be summarized.
            summaries (dict): Summaries for each LLM model.
            device (int): 0 if using GPUs, otherwise -1.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        bart_pipe = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
        bart_output = bart_pipe(article_text, max_length=130, min_length=30)
        summaries["Bart"] = bart_output[0]["summary_text"]
        return summaries

    def summarize_falcons(self, article_text, summaries, device):
        """
        Summarize article using Falcons pretrained transformer

        Args:
            article_text (str): The full text of the article to be summarized.
            summaries (dict): Summaries for each LLM model.
            device (int): 0 if using GPUs, otherwise -1.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        falcons_pipe = pipeline("summarization", model="Falconsai/medical_summarization", device=device)
        falcons_output = falcons_pipe(article_text)
        summaries["Falcons"] = falcons_output[0]["summary_text"]
        return summaries

    def summarize_bigbird(self, article_text, summaries, device):
        """
        Summarize article using BigBird pretrained transformer

        Args:
            article_text (str): The full text of the article to be summarized.
            summaries (dict): Summaries for each LLM model.
            device (int): 0 if using GPUs, otherwise -1.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        bigbird_pipe = pipeline("summarization", model="google/bigbird-pegasus-large-arxiv", device=device)
        bigbird_output = bigbird_pipe(article_text)
        summaries["BigBird"] = bigbird_output[0]["summary_text"]
        return summaries

    def summarize_gpt(self, article_text, system_message, summaries):
        """
        Sends an article to the OpenAI API to generate a summary.

        Args:
            article_text (str): The full text of the article to be summarized.
            system_message (str): Prompt to indicate what the LLM should do.
            summaries (dict): Summaries for each LLM model.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        if self.openai_key != "":
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": article_text},
            ]
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages, stream=False
            )
            summary = response.choices[0].message.content
            summaries["OpenAI"] = summary
            return summaries
        else:
            return summaries

    def summarize_claude(self, article_text, system_message, summaries):
        """
        Sends an article to the Anthropic API to generate a summary.

        Args:
            article_text (str): The full text of the article to be summarized.
            system_message (str): Prompt to indicate what the LLM should do.
            summaries (dict): Summaries for each LLM model.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        if self.anthropic_key != "":
            try:
                response = self.claude.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    system = system_message,
                    messages = [
                        {"role": "user", "content": article_text}
                    ]
                )
                summary = response.content[0].text
                summaries["Anthropic"] = summary
                return summaries
            except Exception as e:
                raise Exception(f"Error summarizing article: {e}")
        else:
            return summaries
        
    def gemini(self, article_text, system_message, summaries):
        """
        Sends an article to the Gemini API to generate a summary.

        Args:
            article_text (str): The full text of the article to be summarized.
            system_message (str): Prompt to indicate what the LLM should do.
            summaries (dict): Summaries for each LLM model.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        if self.gemini_key != "":
            gemini = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                system_instruction=system_message,
            )
            response = gemini.generate_content(article_text)
            summary = response.candidates[0].content.parts[0].text
            summaries["Gemini"] = summary
            return summaries
        else:
            return summaries

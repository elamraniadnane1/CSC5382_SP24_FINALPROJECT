import re
import emoji

def remove_emojis(text):
    """
    Removes emojis from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with emojis removed.
    """
    return emoji.get_emoji_regexp().sub(u'', text)

def extract_emojis(text):
    """
    Extracts all emojis from the text.

    Args:
        text (str): Input text.

    Returns:
        list: A list of emojis found in the text.
    """
    return [char for char in text if char in emoji.UNICODE_EMOJI]

def main():
    sample_text = "Hello World 😊🌍"

    print("Original Text:", sample_text)
    print("Text without Emojis:", remove_emojis(sample_text))
    print("Extracted Emojis:", extract_emojis(sample_text))

if __name__ == "__main__":
    main()

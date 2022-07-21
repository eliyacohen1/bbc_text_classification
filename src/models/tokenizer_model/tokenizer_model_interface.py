import abc


class TokenizerModelInterface:

    @abc.abstractmethod
    def tokenize_text(self, text: str):
        raise NotImplementedError

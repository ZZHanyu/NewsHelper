from transformers import PreTrainedTokenizerFast


class charactors_hander:
    def __init__(self, strs:str) -> None:
        self._raw_str = strs
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-uncased')

        
    def _empty_judge(self) -> bool:
        if len(self._raw_str) <= 0:
            return False
        else:
            return True

    def _lower_charactor(self)->str:
        return self._raw_str.lower()
    
    def _remove_spaces(self)->str:
        for idx in range(len(self._raw_str)):
            if self._raw_str[idx] == " ":
                del self._raw_str[idx] 
        return self._raw_str

    def _split_single_word(self) -> list:
        return self._raw_str.split()
    
    def display_elements(self):
        return self._raw_str
    
    #def _word_tokenizer(self):

    


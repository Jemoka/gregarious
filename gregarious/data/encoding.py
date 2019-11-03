import collections

class BytePairEncoder(object):
    """
    Does bytepair encoding
    """

    @staticmethod
    def __bytepairify(tlist: list) -> list:
        """__bytepairify
        Makes bytepairs out of a list of tokens
        :param tlist: token list
        :type tlist: list
        :rtype: list
        """
        return 

    @staticmethod
    def __init_charlist_generation(sents:list) -> list:
        """__init_charlist_generation
        Makes the initial charlist
        :param sents: sentences
        :type sents: list
        :rtype: list
        """
        product_array = []
        for sent in sents:
            sent = sent+"\0"
            product_array.append(sent)
        oup = []
        for sent in product_array:
            sentList = []
            for word in sent.split(" "):
                sentList = sentList+list(word.lower())
            oup.append(sentList)
        return oup

    @staticmethod
    def __make_bytepair(arr: list) -> list:
        """__make_bytepair
        Makes bytepairs out of an array of tokens
        :param arr: array of tolkens
        :type arr: list
        :rtype: list
        """

        result = [(arr[i], arr[i+1]) for i in range(len(arr)-1)] 
        return result

    @staticmethod
    def __two_counting_dicts_to_one(dicta: dict, dictb: dict) -> dict:
        """__two_counting_dicts_to_one
        Makes two lovely counting dictionaries {key: int count, key: int count, ...} into one lovely counting dictionary
        :param dicta: dictionary a
        :type dicta: dict
        :param dictb: dictionary b
        :type dictb: dict
        :rtype: dict
        """
        oup = collections.defaultdict(int) 
        for key, val in dicta.items():
            oup[key] = oup[key]+val
        for key, val in dictb.items():
            oup[key] = oup[key]+val
        return dict(oup)

    @staticmethod
    def __return_bp_count(arr: list) -> dict:
        """__return_bp_count
        Returns counting dictionary of bytepairs
        :param arr: the bytepairs
        :type arr: list
        :rtype: dict
        """
        bp_count = collections.defaultdict(int)
        for i in arr:
            bp_count[i] += 1
        return dict(bp_count)

    @staticmethod
    def combine(tokens: list, bp: tuple) -> list:
        tk_gen = []
        i = 0
        while i < len(tokens):
            if (tokens[i], tokens[i+1]) == bp:
                tk_gen.append(tokens[i]+tokens[i+1])
                i+=2
            else:
                tk_gen.append(tokens[i])
                i+=1
        return tk_gen
#     @staticmethod
    # def __return_max_bp(bp_dict: dict) -> tuple:
        # return max(bp_dict, key=bp_dict.get)


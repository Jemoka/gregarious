import csv

class DataDescription(object):
    """
    Describes a fields in a CSV
    """

    def __init__(self, header_index:int=0, header_keys_special:dict={}) -> None:
        """__init__

        :param header_index: the index of the header row, usually 0
        :type header_index: int
        :param header_keys_special: {"fileid": "name_in_header", ...}
        :type header_keys_special: dict
        :rtype: None
        """
        header_keys_default = {"handle":"handle", "name": "name", "description":"description", "followers_count":"followers_count", "friends_count":"friends_count", "status":"status", "isBot":"isBot"}
        
        header_keys = {}
        for key, vale in header_keys_default.items():
            hks = header_keys_special.get(str(key))
            if hks:
                header_keys[str(hks)] = str(key)
            else:
                header_keys[str(vale)] = str(key)
        self.header_index = header_index
        self.header_keys = header_keys
        self.ignore_list = []

    def ignore(self, ignore_list:list=None, ignore_str:str=None) -> None:
        assert (ignore_list is not None and ignore_str is None) or (ignore_list is None and ignore_str is not None), "Supply ONLY ignore_list or ignore_str."
        if ignore_str:
            self.ignore_list.append(ignore_list)
        elif ignore_list:
            self.ignore_list = self.ignore_list+ignore_list 
        
class DataFile(object):
    """
    Reads a CSV, gets some fields, serializes them
    """
    
    @staticmethod
    def __optimistically_cast(d):
        """__optimistically_cast
        Cast to the most likely type
        :param d: input 
        """
        
        try:
            res = float(d)
            if res == int(res):
                res = int(res)
        except ValueError:
            if d.upper() == "T" or d.upper() == "TRUE":
                res = True
            elif d.upper() == "F" or d.upper() == "FALSE":
                res = False
            else:
                res = str(d)
        return res

    def __init__(self, directory:str, description:DataDescription) -> None:
        """__init__

        :param directory: the directory of your lovely CSV
        :type directory: str
        :param description: the DataDescription object
        :type description: DataDescription
        :param delimiter: CSV delimiter
        :type delimiter: str
        :param quotechar: CSV quotecahr
        :type quotechar: str
        :rtype: None
        """

        self.dataDescription = description 
        with open(directory, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for indx, line in enumerate(reader):
                if indx == self.dataDescription.header_index:
                    header_items = list(line) 
                    break
            data_raw = {}
            for item in header_items:
                data_raw[item] = []
            for line in reader:
                for indx, item in enumerate(line):
                    data_raw[header_items[indx]].append(self.__optimistically_cast(item))
        self.importedData = {}
        for key, val in data_raw.items():
            if str(key) not in list(self.dataDescription.header_keys.keys()):
                pass
            else:
                self.importedData[self.dataDescription.header_keys[str(key)]] = list(val)
   
    def __encode_p1(self) -> None:
        """__encode_p1
        Turns all chars into charcodes
        :rtype: None
        """
        handles_chars = []
        for handle self.importedData['handle']:
            handle_chars = []
            for char in handle:
                handles_chars.append(ord(char))
            handles_chars.append(handle_chars)

        


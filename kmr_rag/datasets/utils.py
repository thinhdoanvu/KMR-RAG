import os, re
from typing import Optional, Dict, List
from pathlib import Path
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class CRCPlainTextReader(BaseReader):
    """
    Read plain text files and extract text.
    """
    def __init__(
        self
    ) -> None:
        super().__init__()

    def parse_date(self, date, hospID):
        if hospID in {"KMUH"}:
            date_g = re.search(r"(\d+)/(\d+)'(\d+)", date)
            month = date_g.group(1)
            day = date_g.group(2)
            year = date_g.group(3)
            return f"20{year}-{month}-{day}"
        elif hospID in {"CMUH"}:
            return f"2025-09-01"
        elif hospID in {"PPH", "KAFGH"}:
            date_y = int(date[:3])
            month = int(date[3:5])
            day = int(date[5:])
            year = date_y + 1911
            return f"{year}-{month:02d}-{day:02d}"
        
    def preprocess(self, texts, hospID) -> str:
        texts = texts.strip()
        lines = texts.split('\t')
        texts = re.sub(r"^[\da-z]+\t[\da-z]+\s+", "", texts)
        if hospID in {"KMUH"}:
            date = re.search(r"收到(?:日|時間)：\s*([^\s]+)", texts).group(1).strip()
        elif hospID in {"CMUH"}:
            date = "2025-09-01"
        elif hospID in {"PPH", "KAFGH"}:
            date = lines[1].strip()
        date = self.parse_date(date, hospID)
        tmp_text = []
        for text in texts.split('\n'):
            text = text.strip()
            if text:
                tmp_text.append(text) 
        return '\n'.join(tmp_text), date
                
    def load_data(
        self, file: Path, hospID: str, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        
        files = []
        if os.path.isdir(file):
            for path, subdirs, fs in os.walk(file):
                for name in fs:       
                    files.append(os.path.join(path, name))
        # elif os.path.isfile(file):
        #     files = [file]
        else:
            raise Exception("Not support file type")        
        patients = {}
        for file in files:
            with open(file, encoding='utf-8-sig') as txt_file:
                fid = txt_file.name

                text = "\n".join(txt_file.readlines())
                docs = []
                texts = re.split(r"############################################\s+", str(text))
                for doc in texts:
                    if doc != "":
                        text, date = self.preprocess(doc, hospID)          
                        metadata = {
                            "file_path": str(file),
                            "date": date
                        }
                        
                        doc = Document(
                            text=text,
                            metadata=metadata,
                        )
                        docs.append(doc)
                    else:
                        continue    
                patients[fid]=docs                
        return patients
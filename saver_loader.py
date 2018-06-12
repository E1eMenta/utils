import pickle
import numpy as np
import random

class DataSaver:
    def __init__(self, save_path, comment=None):
        self.file = open(save_path, "wb")

        self.comment = comment
        self.path_list = []
        self.discription_list = []

        self.block_num = 0
        self.file.write(np.uint64(self.block_num).tobytes())

        self.current_pos = self.file.tell()


    def save(self, element_list):
        if not all([type(element) is np.ndarray for element in element_list]):
            raise Exception("Wrong data type")

        self.path_list.append(self.current_pos)
        discription = []

        for element in element_list:
            bytes = element.tobytes()
            self.file.write(bytes)

            size = len(bytes)
            discription.append((element.shape, element.dtype, size))

        self.discription_list.append(discription)
        self.current_pos = self.file.tell()

        bytes_annotation = pickle.dumps((self.path_list, self.discription_list, self.comment))
        self.file.write(bytes_annotation)

        self.file.seek(0, 0)
        self.file.write(np.uint64(self.current_pos).tobytes())


        self.file.seek(self.current_pos, 0)


    def close(self):
        self.file.close()


class DataLoader:
    def __init__(self, path, shuffle=False, load_into_memory=False, max_items=None, random_seed=False):
        self.file = open(path, "rb")
        self.file.seek(0, 2)
        self.file_size = self.file.tell()
        self.file.seek(0, 0)

        uint64_size = 8
        annotation_pos = np.fromstring(self.file.read(uint64_size), dtype=np.uint64)[0]

        self.file.seek(annotation_pos, 0)
        bytes_annotation = self.file.read()
        self.path_list, self.discription_list, self.comment = pickle.loads(bytes_annotation)

        if max_items != None:
            self.max_items = max_items if max_items < len(self.path_list) else len(self.path_list)
            self.path_list = self.path_list[:self.max_items]
            self.discription_list = self.path_list[:self.max_items]
        else:
            self.max_items = len(self.path_list)

        self.load_into_memory = load_into_memory
        if load_into_memory:
            self.shuffle = False
            self.index = 0
            self.file.seek(self.path_list[self.index])

            self.data_in_memory = []
            for idx in range(self.max_items):
                self.data_in_memory.append(self.load_from_file())
        
        self.shuffle = shuffle
        if shuffle:
            self.shuffle_data()

        self.index = random.randint(0, self.max_items - 1) if random_seed else 0
        self.file.seek(self.path_list[self.index])


    def getItem(self):
        if self.load_into_memory:
            self.reload_if_needed()
            item = self.data_in_memory[self.index]

            self.index += 1

            return item
        else:
            return self.load_from_file()


    def load_from_file(self):
        self.reload_if_needed()

        if self.shuffle:
            self.file.seek(self.path_list[self.index], 0)


        discriptions = self.discription_list[self.index]
        total_size = sum([size for _, _, size in discriptions])

        bytes_data = self.file.read(total_size)

        item = [None] * len(discriptions)
        current_pos = 0
        for idx, (shape, dtype, size) in enumerate(discriptions):
            bytes_element = bytes_data[current_pos : current_pos + size]
            element = np.fromstring(bytes_element, dtype=dtype)
            item[idx] = np.reshape(element, shape)

            current_pos += size

        self.index += 1

        return item


    def reload_if_needed(self):
        if self.index >= self.max_items:
            self.index = 0
            if self.shuffle:
                self.shuffle_data()


    def __len__(self):
        return self.max_items


    def getComment(self):
        return self.comment


    def shuffle_data(self):
        if self.load_into_memory:
            random.shuffle(self.data_in_memory)
        else:
            combined = list(zip(self.path_list, self.discription_list))
            random.shuffle(combined)

            self.path_list[:], self.discription_list[:] = zip(*combined)
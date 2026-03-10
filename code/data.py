from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, LlavaProcessor, AutoImageProcessor
import pandas as pd
import torch
# from custom_llava import CustomLlavaForConditionalGeneration

# define a dataclass to hold the output of question-image-answer processing
@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor  
    pixel_values: torch.Tensor  
    a_input_ids: torch.Tensor  


# self-defined dataset class for Llava
class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        """
            Initialize the dataset
        Args:
            dataset_dir (str): Path to the dataset directory
        """
        super().__init__()

        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)  # Build the dataset, returning chat data and image directory

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        """
        Build the dataset
        Args:
            data_dir (str): Path to the dataset directory

        Returns:
            Tuple[List[Dict], Path]: List of chat data and image directory path
        """
        data_path = Path(data_dir)   # Convert to Path object
        chat_file = data_path.joinpath("chat.json")  # Chat data file path
        image_dir = data_path.joinpath("images")  # Image directory path
        # Read JSON file and convert to list of dictionaries
        chat_data = pd.read_json(chat_file).to_dict(orient="records")  # [{'id': 'GCC_train_002582585', 'image': 'GCC_train_002582585.jpg', 'conversations': [...]}, ...] 

        return chat_data, image_dir

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        """
        Get the sample at the specified index

        Args:
            index (int): Sample index

        Returns:
            Tuple[str, str, Path]: Tuple containing question text, answer text, and image path
        """
        # Get the current data item
        cur_data = self.chat_data[index]  # {'id': 'GCC_train_002109690', 'image': 'GCC_train_002109690.jpg', 'conversations': [{...}, {...}]}
        # Get the conversation content
        conversations = cur_data.get("conversations")  # [{'from': 'human', 'value': 'Offer a succinct explanation of the picture presented.\n<image>'}, {'from': 'gpt', 'value': "it 's appropriate for teens to want to spend more time with their peers than their parents as they get older ."}]

        # Extract human input (i.e., question) [where <image> is the image placeholder]
        human_input = conversations[0].get("value")   # 'Offer a succinct explanation of the picture presented.\n<image>'
        # Extract standard chatbot output (i.e., answer)
        chatbot_output = conversations[1].get("value")  # "it 's appropriate for teens to want to spend more time with their peers than their parents as they get older ."

        # Construct image path
        image_path = self.image_dir.joinpath(cur_data.get("image"))  
        return human_input, chatbot_output, image_path


def build_qaimage(processor: AutoProcessor, q_text: str, a_text: str, image_path: Path):
    """
    Build QaImageOutput from question, answer, and image path

    Args:
        processor (AutoProcessor): Processor object
        q_text (str): Question text
        a_text (str): Answer text
        image_path (Path): Image path

    Returns:
        QaImageOutput: QaImageOutput object containing input data
    """
    # Build conversation message template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]

     # 应用对话模板生成prompt
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nOffer a succinct explanation of the picture presented.\n<image><|im_end|>\n<|im_start|>assistant\n"

    raw_image = Image.open(image_path)  # Open image file <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=224x224 at 0x7BD9094E4BE0>
    inputs = processor(raw_image, prompt, return_tensors="pt")  # Use processor to process text and image inputs.keys(): dict_keys(['input_ids', 'attention_mask', 'pixel_values'])


    # Process answer text to generate input token ids
    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]  # tensor([[  275,   364,    82,  8311,   369, 26202,   311,  1366,   311,  8329, 803,   882,   448,   862, 25029,  1091,   862,  6562,   438,   807, 633,  9014,   659]]) 

    # Return an object containing all input data
    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )
    return res


class TrainLLavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int) -> None:
        """
        Initialize data collator

        Args:
            processor (AutoProcessor): Processor object
            IGNORE_INDEX (int): Ignore index value (usually -100)
        """
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX

    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
        # pixel_values: torch.Tensor,
    ):
        """
        Convert a single sample to model input format

        Args:
            q_input_ids (torch.Tensor): Token ids of the question
            a_input_ids (torch.Tensor): Token ids of the answer

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Concatenated input token ids and labels
        """
        # Concatenate question, answer, and end token ids
        input_ids = torch.concat(
            [
                q_input_ids,  # Here, 151646 is the token id for <image> → tensor([[151644, 8948, 198, 2610, ..., 624, 151646, 151645, 198, 151644, 77091, 198]])
                a_input_ids,  # tensor([[275, 364, 82, 8311, ..., 9014, 659]])
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        ) # Result: tensor([[151644, 8948, 198, 2610, ..., 624, 151646, 151645, 198, 151644, 77091, 198, 275, 364, 82, 8311, ..., 9014,    659, 151645]])

        # Build labels: fill question part with IGNORE_INDEX, keep answer part unchanged
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ingnore_index),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        ) # Result: tensor([[-100, -100, ..., -100, -100, -100, 275, 364, 82, 8311, ..., 9014, 659, 151645]])

        return input_ids, labels

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        """
        Process batch data. This function is called every time a batch_size of data is taken.

        Args:
            features (List): A list containing a batch
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing a batch_size of data
        """
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            # Build input data for a single sample
            qaimage_output = build_qaimage(
                self.processor, feature[0], feature[1], feature[2]
            )

            # Convert to model input format
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )

            # Record maximum length
            max_input_len_list.append(temp_input_ids.shape[1])  # For example: [53, 59]
            # Save intermediate results
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        # Get the maximum length of each batch
        max_input_len = max(max_input_len_list)

         # Pad input token ids to a uniform length
        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )  # input_ids_list: [tensor([[151644, 8948, ..., 659, 151645]]), tensor([[151644, 8948, 198, 2610, 525, ..., 25956, 151645]])] → final_input_ids: [tensor([[151643, 151643, ..., 151643, 151644, 8948, ..., 659, 151645]]), tensor([[151644, 8948, 198, 2610, 525, ..., 25956, 151645]])]

        # Pad label token ids to a uniform length
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingnore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )  # Similarly obtained as final_input_ids, but the padding symbol is -100, while the padding symbol for final_input_ids is 151643

        # Concatenate image pixel values
        final_pixel_values = torch.concat(pixel_values, axis=0)
        # Build attention mask
        attention_mask = torch.ones_like(final_input_ids)  # tensor([[1, 1, 1, ... 1, 1, 1], [1, 1, 1, 1, 1, 1, ..., 1, 1, 1]])
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0  # tensor([[0, 0, ..., 0, 1, 1, ... 1, 1, 1], [1, 1, ..., 1, 1, 1]])

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }

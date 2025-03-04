#!/usr/bin/env python3

class Llama2FormatChat(object):
    """
        prompt  = f"<<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user_1}"
        prompt += f"<s>[INST] {prompt.strip()} [/INST] {answer_1.strip()} </s>"
        prompt += f"<s>[INST] {user_2.strip()} [/INST] {answer_2.strip()} </s>"
        prompt += f"<s>[INST] {user_3.strip()} [/INST]"
    """
    start_token = "<s>"
    end_token = "</s>"
    sys_start = "<<SYS>>"
    sys_end = "<</SYS>>"
    inst_start = "[INST]"
    inst_end = "[/INST]"
    def __init__(self):
        pass

    def __call__(self, messages) -> str:
        """
        messages are of the form
            messages = [
                {
                    "role": "system",
                    "content": "....",
                },
                {
                    "role": "system",
                    "name": "example_user",
                    "content": "....",
                },
                {
                    "role": "system",
                    "name": "example_assistant",
                    "content": "....",
                },
                {
                    "role": "system",
                    "name": "example_user",
                    "content": "....",
                },
                {
                    "role": "system",
                    "name": "example_assistant",
                    "content": "....",
                },
                {
                    "role": "user",
                    "content": "",
                }
            ]
        """
        # Collect all the system messages
        system_messages = []
        user_messages = []
        assistant_messages = []
        role_names = set()
        for message in messages:
            if message["role"] == "system":
                system_messages.append(message)
                if "name" in message:
                    role_names.add(message["name"])
            elif message["role"] == "user":
                user_messages.append(message)
                if "name" in message:
                    role_names.add(message["name"])
            elif message["role"] == "assistant":
                assistant_messages.append(message)
                if "name" in message:
                    role_names.add(message["name"])
            else:
                raise ValueError(f"Unknown role: {message['role']}")
        sys_prompt = self._format_system_messages(system_messages)
        user_prompt = self._format_user_assistant_messages(user_messages, assistant_messages)
        prompt = \
f"""{sys_prompt}

{user_prompt}"""
        return prompt, role_names

    def _format_system_messages(self, system_message):
        """
        system_message is of the form
        """
        main_system_messages = [sys_msg["content"] for sys_msg in system_message if "name" not in sys_msg]
        main_system_message = "\n".join(main_system_messages)
        example_messages = [(sys_msg["name"], sys_msg["content"]) for sys_msg in system_message if "name" in sys_msg]
        example_messages = [f"`{name}`:\n{msg}" for name, msg in example_messages]
        if len(example_messages) > 0:
            example_message = "\n".join(example_messages)
        else:
            example_message = ""
        if len(example_messages) > 0:
            system_message = \
f"""{main_system_message}

An example of user and assistant interaction is as follows:
{example_message}"""
        else:
            system_message = \
f"""{main_system_message}"""
        system_prompt = \
f"""{Llama2FormatChat.start_token}{Llama2FormatChat.inst_start} {Llama2FormatChat.sys_start}
{system_message}
{Llama2FormatChat.sys_end}"""
        return system_prompt
    
    def _format_user_assistant_messages(self, user_messages, assistant_messages):
        """
        user_messages is of the form
        """
        user_messages = [user_msg["content"] for user_msg in user_messages]
        user_messages = [f"{user_msg} {Llama2FormatChat.inst_end} " for user_msg in user_messages]
        assistant_messages = [assistant_msg["content"] for assistant_msg in assistant_messages]
        assistant_messages = [f"{assistant_msg} {Llama2FormatChat.end_token}{Llama2FormatChat.start_token}{Llama2FormatChat.inst_start} " for assistant_msg in assistant_messages]
        # Combine the messages one after the other
        messages = []
        idx = 0
        while idx < len(user_messages) or idx < len(assistant_messages):
            if idx < len(user_messages):
                messages.append(user_messages[idx])
            if idx < len(assistant_messages):
                messages.append(assistant_messages[idx])
            idx += 1
        message = "".join(messages)
        return message
    
if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        },
        {
            "role": "system",
            "name": "example_assistant", 
            "content": "Our idea seems to be scooped, don't know how to change direction now."
        },
        {
            "role": "user",
            "content": "We changed the direction of the project, but we don't have time to do it.",
        },
        {
            "role": "assistant",
            "content": "Too many changes do not have time to do it.",
        },
        {
            "role": "user",
            "content": "The pot is boiling, probably the water will spill.",
        }
    ]
    llama2_format_chat = Llama2FormatChat()
    prompt, role_names = llama2_format_chat(messages)
    print(prompt)
    print('='*50)
    print(role_names)
    print('-'*100)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        },
        {
            "role": "system",
            "name": "example_assistant", 
            "content": "Our idea seems to be scooped, don't know how to change direction now."
        },
        {
            "role": "user",
            "content": "We changed the direction of the project, but we don't have time to do it.",
        }
    ]
    llama2_format_chat = Llama2FormatChat()
    prompt, role_names = llama2_format_chat(messages)
    print(prompt)
    print('='*50)
    print(role_names)
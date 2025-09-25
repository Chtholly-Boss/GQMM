"""
Harmony chat with tools
"""

import atexit
import argparse
import datetime
import os

try:
    import gnureadline as readline
except ImportError:
    import readline

import torch
import termcolor

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    load_harmony_encoding,
)


REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


def get_user_input():
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        user_input = input()
    else:
        user_input = ""
    user_input_list = [user_input]
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(user_input_list, 0)
    return user_input_list[0]


def main(args):
    from gpt_oss.torch.model import TokenGenerator as TorchGenerator
    from gpt_oss.torch.utils import init_distributed
    device = init_distributed()
    generator = TorchGenerator(args.checkpoint, device)

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    system_message_content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )

    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
    messages = [system_message]

    if args.developer_message:
        developer_message_content = DeveloperContent.new().with_instructions(args.developer_message)
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))
    else:
        developer_message_content = None

    if args.raw:
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation(conversation)
        system_message = encoding.decode(tokens)
        print(system_message, flush=True, end="")
        empty_user_message_tokens = encoding.render(Message.from_role_and_content(Role.USER, ""))
        user_message_start = encoding.decode(empty_user_message_tokens[:-1])
        user_message_end = encoding.decode(empty_user_message_tokens[-1:])
    else:
        # System message
        print(termcolor.colored("System Message:", "cyan"), flush=True)
        print(termcolor.colored("Model Identity:", "cyan"), system_message_content.model_identity, flush=True)
        print(termcolor.colored("Reasoning Effort:", "cyan"), system_message_content.reasoning_effort, flush=True)
        print(termcolor.colored("Conversation Start Date:", "cyan"), system_message_content.conversation_start_date, flush=True)
        print(termcolor.colored("Knowledge Cutoff:", "cyan"), system_message_content.knowledge_cutoff, flush=True)
        if developer_message_content:
            print(termcolor.colored("Developer Message:", "yellow"), flush=True)
            print(developer_message_content.instructions, flush=True)

    # Print the system message and the user message start
    MESSAGE_PADDING = 12
    while True:
        last_message = messages[-1]
        if last_message.recipient is None:
            if args.raw:
                print(user_message_start, end="", flush=True)
                user_message = get_user_input()
                print(user_message_end, flush=True, end="")
            else:
                print(termcolor.colored("User:".ljust(MESSAGE_PADDING), "red"), flush=True)
                user_message = get_user_input()
            user_message = Message.from_role_and_content(Role.USER, user_message)
            messages.append(user_message)
        else:
            raise ValueError(f"Tool or Function call not integrated: {last_message.recipient}")

        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

        if args.raw:
            # Print the last two tokens, which are the start of the assistant message
            print(encoding.decode(tokens[-2:]), flush=True, end="")

        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        field_created = False
        current_output_text = ""
        output_text_delta_buffer = ""
        for predicted_token in generator.generate(tokens, encoding.stop_tokens_for_assistant_actions()):
            parser.process(predicted_token)
            if args.raw:
                print(encoding.decode([predicted_token]), end="", flush=True)
                continue

            if parser.state == StreamState.EXPECT_START:
                print("")  # new line
                field_created = False

            if not parser.last_content_delta:
                continue

            if not field_created:
                field_created = True
                if parser.current_channel == "final":
                    print(termcolor.colored("Assistant:", "green"), flush=True)
                elif parser.current_recipient is not None:
                    print(termcolor.colored(f"Tool call to {parser.current_recipient}:", "cyan"), flush=True)
                else:
                    print(termcolor.colored("CoT:", "yellow"), flush=True)

            should_send_output_text_delta = True
            output_text_delta_buffer += parser.last_content_delta
            if should_send_output_text_delta:
                print(output_text_delta_buffer, end="", flush=True)
                current_output_text += output_text_delta_buffer
                output_text_delta_buffer = ""

        messages += parser.messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
    )
    parser.add_argument(
        "-r",
        "--reasoning-effort",
        metavar="REASONING_EFFORT",
        type=str,
        default="low",
        choices=["high", "medium", "low"],
        help="Reasoning effort",
    )
    parser.add_argument(
        "--developer-message",
        default="",
        help="Developer message",
    )
    parser.add_argument(
        "-c",
        "--context",
        metavar="CONTEXT",
        type=int,
        default=8192,
        help="Max context length",
    )
    parser.add_argument(
        "--raw",
        default=False,
        action="store_true",
        help="Raw mode (does not render Harmony encoding)",
    )
    args = parser.parse_args()

    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        histfile = os.path.join(os.path.expanduser("~"), ".chat")
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(10000)
        except FileNotFoundError:
            pass

        atexit.register(readline.write_history_file, histfile)

    main(args)

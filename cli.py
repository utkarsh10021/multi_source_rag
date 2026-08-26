from rag_service import RAGService

def main():
    service = RAGService()
    print("\nMulti-Source Intelligent RAG Chatbot")
    print("Commands: add URL, add file path, new, clear, quit")

    while True:
        command = input("\nCommand: ").strip()

        if command.lower() == "quit":
            break

        if command.lower() == "new":
            service.reset_knowledge_base()
            print("Knowledge base reset.")
            continue

        if command.lower() == "clear":
            service.clear_memory()
            print("Memory cleared.")
            continue

        if command.startswith("add "):
            target = command[4:].strip()
            try:
                if target.startswith(("http://", "https://")):
                    docs, chunks = service.ingest_website(target)
                else:
                    docs, chunks = service.ingest_file(target)
                print(f"Added {docs} documents and {chunks} chunks.")
            except Exception as exc:
                print(f"Error: {exc}")
            continue

        if service.store.vectorstore is None:
            print("Add a website or file first.")
            continue

        try:
            result = service.ask(command)
            print("\nANSWER:\n", result.answer)
            for source in result.sources:
                print(f"\n[Source {source.number}] {source.title}")
                print(source.source)
        except Exception as exc:
            print(f"Error: {exc}")

if __name__ == "__main__":
    main()

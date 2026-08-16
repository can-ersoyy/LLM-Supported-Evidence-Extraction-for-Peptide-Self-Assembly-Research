from collections import Counter


def process_duplicates(self, items):
        counts = Counter(items)

        unique_items = list(dict.fromkeys(items))

        print("--- Duplicate Report ---")
        duplicate_found = False
        for item, count in counts.items():
            if count > 1:
                duplicate_found = True
                extra_copies = count - 1
                print(
                    f"{item} -> {extra_copies} duplicate(s) found ({count} total occurrences)"
                )

        if not duplicate_found:
            print("No duplicates detected in the list.")

        return unique_items
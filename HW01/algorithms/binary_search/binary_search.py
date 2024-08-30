class BinarySearch:
    def __init__(self) -> None:
        pass

    @staticmethod
    def intersect(p1, p2):
        """
        Perform the merge operation (intersection) between two posting lists.

        Args:
            p1 (list): Posting list of term 1 (sorted list of document IDs).
            p2 (list): Posting list of term 2 (sorted list of document IDs).

        Returns:
            list: A list of document IDs that are present in both p1 and p2.
        """
        answer = []
        i, j = 0, 0

        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                answer.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1

        return answer

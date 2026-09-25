"""Conservative alternate representations; original fields are never replaced."""
import re
import unicodedata

# Only expansions explicitly illustrated in the supplied challenge statement.
LEGAL = {"corp": "corporation", "pvt": "private", "ltd": "limited"}
ADDRESS = {"rd": "road", "st": "street"}


def normalize(value):
    text = unicodedata.normalize("NFKC", value or "").casefold()
    text = text.replace("&", " and ")
    return " ".join(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).replace("_", " ").split())


def expanded(value, field="name"):
    # St can also mean Saint: retain the conservative base as another feature.
    mapping = LEGAL if field == "name" else ADDRESS
    return " ".join(mapping.get(t, t) for t in normalize(value).split())


def accent_fold(value):
    return "".join(c for c in unicodedata.normalize("NFKD", normalize(value))
                   if not unicodedata.combining(c))


def numeric_tokens(value):
    # Preserve leading zeros and all occurrences; numbers are evidence, not vetoes.
    return tuple(re.findall(r"\d+", unicodedata.normalize("NFKC", value or "")))

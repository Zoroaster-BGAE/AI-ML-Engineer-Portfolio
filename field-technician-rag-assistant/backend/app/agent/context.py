from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.agent.schemas import Grade, Understanding
from app.config import Settings
from app.domain.support import ChatSession
from app.llm.base import LLMClient
from app.llm.embed import Embedder
from app.obs.recorder import RunRecorder
from app.retrieval.catalog import Catalog, CatalogDoc
from app.retrieval.repository import Hit, ScopedRepository

# User-facing strings produced by code (never by the model), so they can't be invented or jailbroken.
MSG = {
    "smalltalk": {
        "en": "Hello! Tell me what is happening with {machine}: an alarm code, a symptom, or what you need to do. I will look it up in its manuals.",
        "el": "Γεια σας! Πείτε μου τι συμβαίνει με το {machine}: κωδικός συναγερμού, σύμπτωμα ή τι χρειάζεται να κάνετε. Θα το αναζητήσω στα εγχειρίδιά του.",
    },
    "escalate": {
        # "MACHINE_BRAND" here is the machinery manufacturer, not this product. These
        # strings tell a technician standing at a broken machine who to phone, and
        # the answer is MACHINE_BRAND's service department -- the company that built it.
        # A global rename of the product from MACHINE_BRAND Assistant to Field Technician AI Assistant
        # deliberately left these alone.
        "en": "I could not find this in the manuals for {machine}, and I will not guess where equipment safety is involved. Please contact MACHINE_BRAND service. I have logged the gap so the library can be improved.",
        "el": "Δεν βρήκα αυτή την πληροφορία στα εγχειρίδια για το {machine} και δεν θα κάνω εικασίες όταν εμπλέκεται η ασφάλεια του εξοπλισμού. Παρακαλώ επικοινωνήστε με το τμήμα service της MACHINE_BRAND. Κατέγραψα το κενό ώστε να βελτιωθεί η βιβλιοθήκη.",
    },
    "no_docs": {
        "en": "No documents are indexed for {machine} yet, so I cannot answer from its manuals. Please ask an admin to check the library.",
        "el": "Δεν υπάρχουν ακόμη ευρετηριασμένα έγγραφα για το {machine}, οπότε δεν μπορώ να απαντήσω από τα εγχειρίδιά του. Ζητήστε από έναν διαχειριστή να ελέγξει τη βιβλιοθήκη.",
    },
    "guarded": {
        "en": "I could not produce a safe, grounded answer for this. Please contact MACHINE_BRAND service.",
        "el": "Δεν μπόρεσα να δώσω μια ασφαλή, τεκμηριωμένη απάντηση. Παρακαλώ επικοινωνήστε με το τμήμα service της MACHINE_BRAND.",
    },
    "safety_blocked": {
        "en": "I will not give steps for this, because following them could put you in danger. Please contact MACHINE_BRAND service before working on this part of the machine. I have flagged it for review.",
        "el": "Δεν θα δώσω βήματα για αυτό, γιατί η εκτέλεσή τους θα μπορούσε να σας θέσει σε κίνδυνο. Παρακαλώ επικοινωνήστε με το τμήμα service της MACHINE_BRAND πριν εργαστείτε σε αυτό το μέρος του μηχανήματος. Το έχω επισημάνει για έλεγχο.",
    },
    "safety_header": {"en": "Safety first", "el": "Πρώτα η ασφάλεια"},
    "q_machine": {
        "en": "Which machine are you working on? The site and the model on the nameplate are enough, or send a photo of the nameplate.",
        "el": "Σε ποιο μηχάνημα εργάζεστε; Αρκεί ο χώρος και το μοντέλο από την πινακίδα, ή στείλτε μια φωτογραφία της πινακίδας.",
    },
    "none_of_these": {"en": "None of these", "el": "Κανένα από αυτά"},
    "machine_locked": {
        "en": "Got it: {machine}. What is happening with it? An alarm code, a symptom or the job you need to do.",
        "el": "Εντάξει: {machine}. Τι συμβαίνει με αυτό; Ένας κωδικός συναγερμού, ένα σύμπτωμα ή η εργασία που χρειάζεται να κάνετε.",
    },
    "not_sure": {"en": "Not sure", "el": "Δεν ξέρω"},
    "q_component": {"en": "Which part of the machine is it?", "el": "Ποιο μέρος του μηχανήματος αφορά;"},
    "q_document": {"en": "Which document set applies to your machine?", "el": "Ποιο σετ εγγράφων αντιστοιχεί στο μηχάνημά σας;"},
    "q_doc_type": {"en": "What kind of document do you need?", "el": "Τι είδους έγγραφο χρειάζεστε;"},
    "q_symptom": {"en": "What is the machine doing?", "el": "Τι κάνει το μηχάνημα;"},
}


def msg(key: str, lang: str, **fmt) -> str:
    table = MSG[key]
    return (table.get(lang) or table["en"]).format(**fmt)


@dataclass
class TurnContext:
    db: Session
    llm: LLMClient
    embedder: Embedder
    repo: ScopedRepository
    rec: RunRecorder
    session: ChatSession
    settings: Settings
    machine_label: str
    memory: dict
    user_text: str
    prev_issue: str | None = None
    catalog: Catalog | None = None
    understanding: Understanding | None = None
    plan: list[CatalogDoc] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    grade: Grade | None = None
    expand_count: int = 0
    images: list = field(default_factory=list)  # ImageInput photos sent with this turn

    @property
    def lang(self) -> str:
        return self.understanding.language if self.understanding else "en"

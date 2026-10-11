"""Fixed bilingual texts of the chatbot: greeting, refusals, notes, and the "my field" templates.

Every number in a template is filled from Member 2 / Member 3 API values at run time; the templates contain no
agronomic numbers of their own. The Sinhala wording was drafted by the AI coding assistant and is unverified
(SINHALA_VERIFIED = False, sent to the frontend so the officer view can say so).
"""

from __future__ import annotations

SINHALA_VERIFIED = False

GREETING = {
    "en": "Hello! I am SoilMind. Ask me about paddy fertilizer, pests and diseases, weeds, water, land preparation "
          "or rice varieties. Ask \"how is my field?\" to hear what your sensors say.",
    "si": "ආයුබෝවන්! මම SoilMind. වී වගාවේ පොහොර, පළිබෝධ හා රෝග, වල් පැලෑටි, ජලය, බිම් සැකසීම හෝ වී ප්‍රභේද ගැන "
          "මගෙන් අසන්න. ඔබේ සංවේදක පවසන දේ දැනගැනීමට \"මගේ කුඹුර කොහොමද?\" කියා අසන්න.",
}

REFUSE_OUT_OF_SCOPE = {
    "en": "Sorry, I can only answer questions about paddy cultivation and your field. I have forwarded your "
          "question to an agriculture officer (reference {ref}).",
    "si": "කණගාටුයි, මට පිළිතුරු දිය හැක්කේ වී වගාව සහ ඔබේ කුඹුර පිළිබඳ ප්‍රශ්නවලට පමණයි. ඔබේ ප්‍රශ්නය "
          "කෘෂිකර්ම නිලධාරියෙකුට යොමු කළා (යොමු අංකය {ref}).",
}

REFUSE_NOT_FOUND = {
    "en": "I could not find this in the Department of Agriculture documents I have, so I will not guess. I have "
          "forwarded your question to an agriculture officer (reference {ref}).",
    "si": "මා සතු කෘෂිකර්ම දෙපාර්තමේන්තු ලේඛනවල මෙයට පිළිතුරක් හමු නොවූ නිසා මම අනුමාන නොකරමි. ඔබේ ප්‍රශ්නය "
          "කෘෂිකර්ම නිලධාරියෙකුට යොමු කළා (යොමු අංකය {ref}).",
}

# shown above the copied sentences when the best passage is in the other language
OTHER_LANGUAGE_NOTE = {
    "en": "The Department of Agriculture text for this is in Sinhala:",
    "si": "මේ සඳහා කෘෂිකර්ම දෙපාර්තමේන්තු පාඨය ඇත්තේ ඉංග්‍රීසියෙනි:",
}

FROM_DOCUMENTS = {
    "en": "From the Department of Agriculture documents:",
    "si": "කෘෂිකර්ම දෙපාර්තමේන්තු ලේඛනවලින්:",
}

LIVE_UNAVAILABLE = {
    "en": "I could not read your field's live data just now, so I cannot say anything about your own plot.",
    "si": "ඔබේ කුඹුරේ සජීවී දත්ත දැන් කියවිය නොහැකි වූ නිසා ඔබේ කුඹුර ගැන කිසිවක් කිව නොහැක.",
}

NO_RECENT_READINGS = {
    "en": "Your field's sensors have sent no recent readings (last reading: {time}), so I cannot tell you how your "
          "field is now. Please check the sensor nodes.",
    "si": "ඔබේ කුඹුරේ සංවේදක වලින් මෑතකදී කියවීම් ලැබී නැත (අවසන් කියවීම: {time}), එබැවින් ඔබේ කුඹුර දැන් කොහොමදැයි "
          "කිව නොහැක. කරුණාකර සංවේදක පරීක්ෂා කරන්න.",
}

LIVE_SOURCE = {
    "en": "SoilMind live data (sensors, health score and advice)",
    "si": "SoilMind සජීවී දත්ත (සංවේදක, සෞඛ්‍ය ලකුණු සහ උපදෙස්)",
}

GRADE = {
    "good": {"en": "good", "si": "හොඳයි"},
    "watch": {"en": "watch", "si": "අවධානයෙන් සිටින්න"},
    "act": {"en": "act now", "si": "දැන් ක්‍රියා කරන්න"},
}

VARIABLE = {
    "n": {"en": "nitrogen", "si": "නයිට්‍රජන්"},
    "p": {"en": "phosphorus", "si": "පොස්පරස්"},
    "k": {"en": "potassium", "si": "පොටෑසියම්"},
    "ph": {"en": "pH", "si": "pH"},
    "moisture": {"en": "moisture", "si": "තෙතමනය"},
    "temp": {"en": "soil temperature", "si": "පසේ උෂ්ණත්වය"},
    "ec": {"en": "EC", "si": "EC"},
}

UNIT = {"n": "mg/kg", "p": "mg/kg", "k": "mg/kg", "ph": "", "moisture": "%", "temp": "°C", "ec": "dS/m"}

FIELD_STATUS = {
    "en": "Your field ({plot}) health score is {score} out of 100: {grade}. The weakest point is {var} at {node} "
          "({value}{unit}). Field averages: nitrogen {n} mg/kg, phosphorus {p} mg/kg, potassium {k} mg/kg, "
          "pH {ph}, moisture {moisture}%. Last reading: {time}.",
    "si": "ඔබේ කුඹුරේ ({plot}) සෞඛ්‍ය ලකුණු 100න් {score}: {grade}. දුර්වලම ස්ථානය {node} හි {var} "
          "({value}{unit}). කුඹුරේ සාමාන්‍ය අගයන්: නයිට්‍රජන් {n} mg/kg, පොස්පරස් {p} mg/kg, පොටෑසියම් {k} mg/kg, "
          "pH {ph}, තෙතමනය {moisture}%. අවසන් කියවීම: {time}.",
}

ALERTS_LINE = {
    "en": "Alerts: {titles}.",
    "si": "අනතුරු ඇඟවීම්: {titles}.",
}

NEXT_ACTION = {
    "en": "For your field ({plot}, {area} acres): {instruction} Do it between {start} and {end}. "
          "({per_acre} kg per acre; {reason})",
    "si": "ඔබේ කුඹුර සඳහා ({plot}, අක්කර {area}): {instruction} {start} සිට {end} දක්වා කාලයේදී කරන්න. "
          "(අක්කරයකට කි.ග්‍රෑ. {per_acre}; {reason})",
}

NEXT_ACTION_ONE_DAY = {
    "en": "For your field ({plot}, {area} acres): {instruction} Do it on {start}. ({per_acre} kg per acre; {reason})",
    "si": "ඔබේ කුඹුර සඳහා ({plot}, අක්කර {area}): {instruction} {start} දින කරන්න. "
          "(අක්කරයකට කි.ග්‍රෑ. {per_acre}; {reason})",
}

NO_ACTION = {
    "en": "For your field: no {product} application is planned in the current advice.",
    "si": "ඔබේ කුඹුර සඳහා: වත්මන් උපදෙස් අනුව {product} යෙදීමක් සැලසුම් කර නැත.",
}

# Words that tell which product a question is about (English and Sinhala)
PRODUCT_WORDS = {
    "urea": ["urea", "nitrogen", "යූරියා", "නයිට්‍රජන්", "yuriya"],
    "TSP": ["tsp", "phosph", "ටී.එස්.පී", "පොස්පරස්"],
    "MOP": ["mop", "potash", "potassium", "එම්.ඕ.පී", "පොටෑසියම්"],
    "dolomite": ["lime", "dolomite", "ph", "acid", "හුණු", "ඩොලමයිට්", "ආම්ලික"],
}


# ---- Phase 7: answers about one zone of the field, and "why this recommendation?" --------------------------------
# (filled from /api/m3/alerts and /api/m2/recommendations; no numbers of their own)

ZONE_ALERT = {
    "en": "{zone} part of your field: {title}. {body}",
    "si": "ඔබේ කුඹුරේ {zone} කොටස: {title}. {body}",
}

ZONE_NO_ALERT = {
    "en": "There is no alert for the {zone} part of your field right now.",
    "si": "මේ මොහොතේ ඔබේ කුඹුරේ {zone} කොටස සඳහා අනතුරු ඇඟවීමක් නැත.",
}

ACTION_WHAT = {
    "en": "{title} ({plot}): {instruction} Do it between {start} and {end} ({per_acre} kg per acre).",
    "si": "{title} ({plot}): {instruction} {start} සිට {end} දක්වා කාලයේදී කරන්න (අක්කරයකට කි.ග්‍රෑ. {per_acre}).",
}

ACTION_WHAT_ONE_DAY = {
    "en": "{title} ({plot}): {instruction} Do it on {start} ({per_acre} kg per acre).",
    "si": "{title} ({plot}): {instruction} {start} දින කරන්න (අක්කරයකට කි.ග්‍රෑ. {per_acre}).",
}

ACTION_WHY = {
    "en": "Why: {reason}",
    "si": "හේතුව: {reason}",
}

ACTION_WHERE = {
    "en": "Where: {zone} ({area} acres; sensors {nodes}).",
    "si": "කොතැනද: {zone} (අක්කර {area}; සංවේදක {nodes}).",
}

ACTION_EFFECT = {
    "en": "Expected effect: {var} {change}{unit} by {date}, from the soil forecast model.",
    "si": "අපේක්ෂිත බලපෑම: {date} වන විට {var} {change}{unit} (පාංශු පුරෝකථන ආකෘතියෙන්).",
}

ACTION_GONE = {
    "en": "That recommendation is no longer in the current advice (the advice changes when the readings change).",
    "si": "එම නිර්දේශය වත්මන් උපදෙස්වල තවදුරටත් නැත (කියවීම් වෙනස් වන විට උපදෙස් වෙනස් වේ).",
}

ADVICE_SOURCE = {
    "en": "SoilMind advice (RRDI / DOA sources)",
    "si": "SoilMind උපදෙස් (RRDI / DOA මූලාශ්‍ර)",
}

# Shown when one of our models failed and a simpler step answered instead (graceful degradation).
DEGRADED_NOTE = {
    "en": "The sentence picker is not available right now, so these are the closest sentences of the best "
          "matching Department of Agriculture passage.",
    "si": "වාක්‍ය තෝරන ආකෘතිය දැන් ලබාගත නොහැකි නිසා, වඩාත් ගැළපෙන කෘෂිකර්ම දෙපාර්තමේන්තු ඡේදයේ ආසන්නතම "
          "වාක්‍ය මෙසේය.",
}

# the product that corrects a low nutrient (the products Member 2's RRDI-sourced advice uses for each)
VARIABLE_PRODUCT = {"n": "urea", "p": "TSP", "k": "MOP", "ph": "dolomite"}

# Words that name a part of the field in a question (compass zones, as /api/m3/alerts names them)
ZONE_WORDS = {
    "north-west": ["north-west", "north west", "northwest", "වයඹ"],
    "north-east": ["north-east", "north east", "northeast", "ඊසාන"],
    "south-west": ["south-west", "south west", "southwest", "නිරිත"],
    "south-east": ["south-east", "south east", "southeast", "ගිනිකොන"],
    "north": ["north part", "northern", "උතුරු"],
    "south": ["south part", "southern", "දකුණු"],
    "east": ["east part", "eastern", "නැගෙනහිර"],
    "west": ["west part", "western", "බටහිර"],
    "centre": ["centre", "center", "middle", "මැද"],
}

# ---- Phase 7: the questions the cross-links pre-fill (POST /api/m4/prefill) ------------------------------------
# Worded so our intent model routes them to the "my field" answer (tests/test_m4_crosslinks.py checks every one);
# the context sent with them makes the answer about exactly that zone or recommendation.

ZONE_NAME = {
    "north-west": {"en": "north-west", "si": "වයඹ"}, "north-east": {"en": "north-east", "si": "ඊසාන"},
    "south-west": {"en": "south-west", "si": "නිරිත"}, "south-east": {"en": "south-east", "si": "ගිනිකොන"},
    "north": {"en": "north", "si": "උතුරු"}, "south": {"en": "south", "si": "දකුණු"},
    "east": {"en": "east", "si": "නැගෙනහිර"}, "west": {"en": "west", "si": "බටහිර"},
    "centre": {"en": "middle", "si": "මැද"},
}

PREFILL_ZONE_LOW = {  # a nutrient below its deficiency level in one part of the field
    "en": "Why is {variable} low in the {zone} part of my field?",
    "si": "මගේ කුඹුරේ {zone} කොටසේ තත්ත්වය කුමක්ද?",
}

PREFILL_ZONE = {  # any other problem in one part of the field
    "en": "How is the {zone} part of my field doing, and what should I do?",
    "si": "මගේ කුඹුරේ {zone} කොටසේ තත්ත්වය කුමක්ද?",
}

PREFILL_FIELD = {  # no zone: the whole field
    "en": "How is my field today?",
    "si": "මගේ කුඹුර අද කොහොමද?",
}

PREFILL_ACTION = {
    "en": "Why should I apply {product} on my field now?",
    "si": "මගේ කුඹුරට දැන් {product} දැමිය යුත්තේ ඇයි?",
}

PRODUCT_NAME = {
    "urea": {"en": "urea", "si": "යූරියා"}, "TSP": {"en": "TSP", "si": "TSP"},
    "MOP": {"en": "MOP", "si": "MOP"}, "dolomite": {"en": "dolomite", "si": "ඩොලමයිට්"},
}


def prefill(context: dict | None, lang: str) -> str:
    """The question a cross-link puts in the chat box for this context, in the UI language."""
    c = context or {}
    if c.get("kind") == "recommendation":
        name = PRODUCT_NAME.get(c.get("product") or "", {}).get(lang) or c.get("product") or ""
        return PREFILL_ACTION[lang].format(product=name)
    zone = ZONE_NAME.get(c.get("zone") or "")
    if c.get("kind") == "zone" and zone:
        var = c.get("variable")
        if var in ("n", "p", "k", "ph") and c.get("side") != "high":
            return PREFILL_ZONE_LOW[lang].format(variable=VARIABLE[var][lang], zone=zone[lang])
        return PREFILL_ZONE[lang].format(zone=zone[lang])
    return PREFILL_FIELD[lang]


def prefill_label(context: dict | None) -> dict[str, str]:
    """What a cross-link is about, in both languages (zone and layer, or the product), for the chat's chip."""
    c = context or {}
    if c.get("kind") == "recommendation":
        return {lg: PRODUCT_NAME.get(c.get("product") or "", {}).get(lg) or (c.get("product") or "") for lg in ("en", "si")}
    zone = ZONE_NAME.get(c.get("zone") or "")
    if c.get("kind") == "zone" and zone:
        var = VARIABLE.get(c.get("variable") or "", {})
        nodes = ", ".join(c.get("nodes") or [])
        return {lg: " · ".join(x for x in (zone[lg], var.get(lg, ""), nodes) if x) for lg in ("en", "si")}
    return {"en": "my field", "si": "මගේ කුඹුර"}

"""
Constants and configuration parameters for Student Performance Prediction

This module contains all the constants, data type definitions, and configuration parameters
used throughout the feature engineering pipeline for predicting student performance from
gameplay data. The constants are organized into logical groups for easy maintenance
and understanding of the feature engineering process.

Key Components:
- Data type definitions for Polars DataFrame optimization
- Categorical and numerical column specifications
- Feature engineering lists for different game elements
- Question categorization for different modeling approaches
- Training configuration parameters
- Level group boundaries for temporal analysis

Author: Competition Team
Competition: Predict Student Performance from Game Play
Position: 35/2051 (Silver Medal)
"""

# Data path configuration
DATA_PATH = "../data/predict_student_performance_from_gameplay"

# Data types for Polars DataFrame optimization
# These types are optimized for memory efficiency and performance
DTYPES = {
    "session_id": "Int64",  # Unique session identifier for each student
    "elapsed_time": "Int64",  # Time elapsed since session start (milliseconds)
    "event_name": "Categorical",  # Type of user interaction event
    "name": "Categorical",  # Name/identifier of the interactive element
    "page": "Float32",  # Page number in multi-page content
    "room_coor_x": "Float32",  # X-coordinate within the game room
    "room_coor_y": "Float32",  # Y-coordinate within the game room
    "screen_coor_x": "Float32",  # X-coordinate on the screen
    "screen_coor_y": "Float32",  # Y-coordinate on the screen
    "hover_duration": "Float32",  # Duration of hover interaction
    "text": "Categorical",  # Text content displayed to the user
    "fqid": "Categorical",  # Fully qualified ID for game elements
    "room_fqid": "Categorical",  # Room-specific fully qualified ID
    "text_fqid": "Categorical",  # Text-specific fully qualified ID
    "fullscreen": "Float32",  # Fullscreen mode indicator
    "hq": "Float32",  # High quality setting indicator
    "music": "Float32",  # Music setting indicator
    "level_group": "Categorical",  # Game level grouping (0-4, 5-12, 13-22)
}

# Categorical columns for feature engineering
# These columns contain discrete values that need encoding or aggregation
CATS = ["event_name", "name", "fqid", "room_fqid", "text_fqid"]

# Numerical columns for feature engineering
# These columns contain continuous values for statistical aggregation
NUMS = [
    "page",  # Page numbers for multi-page content
    "room_coor_x",  # Room X-coordinates for spatial analysis
    "room_coy_y",  # Room Y-coordinates for spatial analysis (note: typo in original)
    "screen_coor_x",  # Screen X-coordinates for interaction analysis
    "screen_coor_y",  # Screen Y-coordinates for interaction analysis
    "hover_duration",  # Hover time for engagement measurement
    "elapsed_time_diff",  # Time difference between consecutive events
]

# Name features for aggregation
# Common interactive element names that appear frequently in the game
NAME_FEATURE = ["basic", "undefined", "close", "open", "prev", "next"]

# Event name features for aggregation
# Types of user interactions that are tracked and analyzed
EVENT_NAME_FEATURE = [
    "cutscene_click",  # Clicking during cutscenes
    "person_click",  # Clicking on person characters
    "navigate_click",  # Navigation-related clicks
    "observation_click",  # Observation-based interactions
    "notification_click",  # Clicking on notifications
    "object_click",  # Clicking on game objects
    "object_hover",  # Hovering over objects
    "map_hover",  # Hovering over map elements
    "map_click",  # Clicking on map elements
    "checkpoint",  # Reaching game checkpoints
    "notebook_click",  # Interacting with notebook elements
]

# FQID lists for feature engineering
# Fully Qualified IDs represent specific game locations and interactions
# These are used to create location-specific features and analyze player movement patterns
FQID_LISTS = [
    "worker",  # Worker character interactions
    "archivist",  # Archivist character interactions
    "gramps",  # Grandfather character interactions
    "wells",  # Wells character interactions
    "toentry",  # Entry point interactions
    "confrontation",  # Confrontation scene interactions
    "crane_ranger",  # Crane ranger interactions
    "groupconvo",  # Group conversation interactions
    "flag_girl",  # Flag girl character interactions
    "tomap",  # Map navigation interactions
    "tostacks",  # Stacks area interactions
    "tobasement",  # Basement area interactions
    "archivist_glasses",  # Archivist with glasses interactions
    "boss",  # Boss character interactions
    "journals",  # Journal interactions
    "seescratches",  # Scratch observation interactions
    "groupconvo_flag",  # Flag-related group conversations
    "cs",  # Cutscene interactions
    "teddy",  # Teddy bear interactions
    "expert",  # Expert character interactions
    "businesscards",  # Business card interactions
    "ch3start",  # Chapter 3 start interactions
    "tunic.historicalsociety",  # Historical society tunic interactions
    "tofrontdesk",  # Front desk navigation
    "savedteddy",  # Teddy rescue interactions
    "plaque",  # Plaque interactions
    "glasses",  # Glasses-related interactions
    "tunic.drycleaner",  # Dry cleaner tunic interactions
    "reader_flag",  # Reader flag interactions
    "tunic.library",  # Library tunic interactions
    "tracks",  # Track-related interactions
    "tunic.capitol_2",  # Capitol 2 tunic interactions
    "trigger_scarf",  # Scarf trigger interactions
    "reader",  # Reader interactions
    "directory",  # Directory interactions
    "tunic.capitol_1",  # Capitol 1 tunic interactions
    "journals.pic_0.next",  # Journal picture navigation
    "unlockdoor",  # Door unlocking interactions
    "tunic",  # General tunic interactions
    "what_happened",  # What happened interactions
    "tunic.kohlcenter",  # Kohl center tunic interactions
    "tunic.humanecology",  # Human ecology tunic interactions
    "colorbook",  # Color book interactions
    "logbook",  # Log book interactions
    "businesscards.card_0.next",  # Business card navigation
    "journals.hub.topics",  # Journal topic hub interactions
    "logbook.page.bingo",  # Log book bingo interactions
    "journals.pic_1.next",  # Journal picture 1 navigation
    "journals_flag",  # Journal flag interactions
    "reader.paper0.next",  # Reader paper navigation
    "tracks.hub.deer",  # Deer track hub interactions
    "reader_flag.paper0.next",  # Reader flag paper navigation
    "trigger_coffee",  # Coffee trigger interactions
    "wellsbadge",  # Wells badge interactions
    "journals.pic_2.next",  # Journal picture 2 navigation
    "tomicrofiche",  # Microfiche interactions
    "journals_flag.pic_0.bingo",  # Journal flag bingo
    "plaque.face.date",  # Plaque face date interactions
    "notebook",  # Notebook interactions
    "tocloset_dirty",  # Dirty closet navigation
    "businesscards.card_bingo.bingo",  # Business card bingo
    "businesscards.card_1.next",  # Business card 1 navigation
    "tunic.wildlife",  # Wildlife tunic interactions
    "tunic.hub.slip",  # Hub slip tunic interactions
    "tocage",  # Cage navigation
    "journals.pic_2.bingo",  # Journal picture 2 bingo
    "tocollectionflag",  # Collection flag navigation
    "tocollection",  # Collection navigation
    "chap4_finale_c",  # Chapter 4 finale interactions
    "chap2_finale_c",  # Chapter 2 finale interactions
    "lockeddoor",  # Locked door interactions
    "journals_flag.hub.topics",  # Journal flag topic hub
    "tunic.capitol_0",  # Capitol 0 tunic interactions
    "reader_flag.paper2.bingo",  # Reader flag paper 2 bingo
    "photo",  # Photo interactions
    "tunic.flaghouse",  # Flag house tunic interactions
    "reader.paper1.next",  # Reader paper 1 navigation
    "directory.closeup.archivist",  # Directory closeup archivist
    "intro",  # Introduction interactions
    "businesscards.card_bingo.next",  # Business card bingo navigation
    "reader.paper2.bingo",  # Reader paper 2 bingo
    "retirement_letter",  # Retirement letter interactions
    "remove_cup",  # Cup removal interactions
    "journals_flag.pic_0.next",  # Journal flag picture 0 navigation
    "magnify",  # Magnification interactions
    "coffee",  # Coffee interactions
    "key",  # Key interactions
    "togrampa",  # Grandfather navigation
    "reader_flag.paper1.next",  # Reader flag paper 1 navigation
    "janitor",  # Janitor interactions
    "tohallway",  # Hallway navigation
    "chap1_finale",  # Chapter 1 finale interactions
    "report",  # Report interactions
    "outtolunch",  # Out to lunch interactions
    "journals_flag.hub.topics_old",  # Old journal flag topic hub
    "journals_flag.pic_1.next",  # Journal flag picture 1 navigation
    "reader.paper2.next",  # Reader paper 2 navigation
    "chap1_finale_c",  # Chapter 1 finale C interactions
    "reader_flag.paper2.next",  # Reader flag paper 2 navigation
    "door_block_talk",  # Door block talk interactions
    "journals_flag.pic_1.bingo",  # Journal flag picture 1 bingo
    "journals_flag.pic_2.next",  # Journal flag picture 2 navigation
    "journals_flag.pic_2.bingo",  # Journal flag picture 2 bingo
    "block_magnify",  # Block magnify interactions
    "reader.paper0.prev",  # Reader paper 0 previous
    "block",  # Block interactions
    "reader_flag.paper0.prev",  # Reader flag paper 0 previous
    "block_0",  # Block 0 interactions
    "door_block_clean",  # Door block clean interactions
    "reader.paper2.prev",  # Reader paper 2 previous
    "reader.paper1.prev",  # Reader paper 1 previous
    "doorblock",  # Door block interactions
    "tocloset",  # Closet navigation
    "reader_flag.paper2.prev",  # Reader flag paper 2 previous
    "reader_flag.paper1.prev",  # Reader flag paper 1 previous
    "block_tomap2",  # Block to map 2 interactions
    "journals_flag.pic_0_old.next",  # Old journal flag picture 0 navigation
    "journals_flag.pic_1_old.next",  # Old journal flag picture 1 navigation
    "block_tocollection",  # Block to collection interactions
    "block_nelson",  # Block nelson interactions
    "journals_flag.pic_2_old.next",  # Old journal flag picture 2 navigation
    "block_tomap1",  # Block to map 1 interactions
    "block_badge",  # Block badge interactions
    "need_glasses",  # Need glasses interactions
    "block_badge_2",  # Block badge 2 interactions
    "fox",  # Fox interactions
    "block_1",  # Block 1 interactions
]

# Text lists for feature engineering
# Specific text content that appears in the game, used for text-based feature creation
# These texts are analyzed for frequency and timing to understand student engagement
TEXT_LISTS = [
    "tunic.historicalsociety.cage.confrontation",  # Historical society cage confrontation text
    "tunic.wildlife.center.crane_ranger.crane",  # Wildlife center crane ranger text
    "tunic.historicalsociety.frontdesk.archivist.newspaper",  # Archivist newspaper text
    "tunic.historicalsociety.entry.groupconvo",  # Entry group conversation text
    "tunic.wildlife.center.wells.nodeer",  # Wells no deer text
    "tunic.historicalsociety.frontdesk.archivist.have_glass",  # Archivist have glass text
    "tunic.drycleaner.frontdesk.worker.hub",  # Dry cleaner worker hub text
    "tunic.historicalsociety.closet_dirty.gramps.news",  # Dirty closet gramps news text
    "tunic.humanecology.frontdesk.worker.intro",  # Human ecology worker intro text
    "tunic.historicalsociety.frontdesk.archivist_glasses.confrontation",  # Archivist glasses confrontation text
    "tunic.historicalsociety.basement.seescratches",  # Basement see scratches text
    "tunic.historicalsociety.collection.cs",  # Collection cutscene text
    "tunic.flaghouse.entry.flag_girl.hello",  # Flag house flag girl hello text
    "tunic.historicalsociety.collection.gramps.found",  # Collection gramps found text
    "tunic.historicalsociety.basement.ch3start",  # Basement chapter 3 start text
    "tunic.historicalsociety.entry.groupconvo_flag",  # Entry group conversation flag text
    "tunic.library.frontdesk.worker.hello",  # Library worker hello text
    "tunic.library.frontdesk.worker.wells",  # Library worker wells text
    "tunic.historicalsociety.collection_flag.gramps.flag",  # Collection flag gramps flag text
    "tunic.historicalsociety.basement.savedteddy",  # Basement saved teddy text
    "tunic.library.frontdesk.worker.nelson",  # Library worker nelson text
    "tunic.wildlife.center.expert.removed_cup",  # Wildlife expert removed cup text
    "tunic.library.frontdesk.worker.flag",  # Library worker flag text
    "tunic.historicalsociety.frontdesk.archivist.hello",  # Archivist hello text
    "tunic.historicalsociety.closet.gramps.intro_0_cs_0",  # Closet gramps intro cutscene text
    "tunic.historicalsociety.entry.boss.flag",  # Entry boss flag text
    "tunic.flaghouse.entry.flag_girl.symbol",  # Flag house flag girl symbol text
    "tunic.historicalsociety.closet_dirty.trigger_scarf",  # Dirty closet trigger scarf text
    "tunic.drycleaner.frontdesk.worker.done",  # Dry cleaner worker done text
    "tunic.historicalsociety.closet_dirty.what_happened",  # Dirty closet what happened text
    "tunic.wildlife.center.wells.animals",  # Wildlife wells animals text
    "tunic.historicalsociety.closet.teddy.intro_0_cs_0",  # Closet teddy intro cutscene text
    "tunic.historicalsociety.cage.glasses.afterteddy",  # Cage glasses after teddy text
    "tunic.historicalsociety.cage.teddy.trapped",  # Cage teddy trapped text
    "tunic.historicalsociety.cage.unlockdoor",  # Cage unlock door text
    "tunic.historicalsociety.stacks.journals.pic_2.bingo",  # Stacks journals picture 2 bingo text
    "tunic.historicalsociety.entry.wells.flag",  # Entry wells flag text
    "tunic.humanecology.frontdesk.worker.badger",  # Human ecology worker badger text
    "tunic.historicalsociety.stacks.journals_flag.pic_0.bingo",  # Stacks journals flag picture 0 bingo text
    "tunic.historicalsociety.closet.intro",  # Closet intro text
    "tunic.historicalsociety.closet.retirement_letter.hub",  # Closet retirement letter hub text
    "tunic.historicalsociety.entry.directory.closeup.archivist",  # Entry directory closeup archivist text
    "tunic.historicalsociety.collection.tunic.slip",  # Collection tunic slip text
    "tunic.kohlcenter.halloffame.plaque.face.date",  # Kohl center hall of fame plaque date text
    "tunic.historicalsociety.closet_dirty.trigger_coffee",  # Dirty closet trigger coffee text
    "tunic.drycleaner.frontdesk.logbook.page.bingo",  # Dry cleaner logbook page bingo text
    "tunic.library.microfiche.reader.paper2.bingo",  # Library microfiche reader paper 2 bingo text
    "tunic.kohlcenter.halloffame.togrampa",  # Kohl center hall of fame to grampa text
    "tunic.capitol_2.hall.boss.haveyougotit",  # Capitol 2 hall boss have you got it text
    "tunic.wildlife.center.wells.nodeer_recap",  # Wildlife wells no deer recap text
    "tunic.historicalsociety.cage.glasses.beforeteddy",  # Cage glasses before teddy text
    "tunic.historicalsociety.closet_dirty.gramps.helpclean",  # Dirty closet gramps help clean text
    "tunic.wildlife.center.expert.recap",  # Wildlife expert recap text
    "tunic.historicalsociety.frontdesk.archivist.have_glass_recap",  # Archivist have glass recap text
    "tunic.historicalsociety.stacks.journals_flag.pic_1.bingo",  # Stacks journals flag picture 1 bingo text
    "tunic.historicalsociety.cage.lockeddoor",  # Cage locked door text
    "tunic.historicalsociety.stacks.journals_flag.pic_2.bingo",  # Stacks journals flag picture 2 bingo text
    "tunic.historicalsociety.collection.gramps.lost",  # Collection gramps lost text
    "tunic.historicalsociety.closet.notebook",  # Closet notebook text
    "tunic.historicalsociety.frontdesk.magnify",  # Front desk magnify text
    "tunic.humanecology.frontdesk.businesscards.card_bingo.bingo",  # Human ecology business cards bingo text
    "tunic.wildlife.center.remove_cup",  # Wildlife remove cup text
    "tunic.library.frontdesk.wellsbadge.hub",  # Library wells badge hub text
    "tunic.wildlife.center.tracks.hub.deer",  # Wildlife tracks hub deer text
    "tunic.historicalsociety.frontdesk.key",  # Front desk key text
    "tunic.library.microfiche.reader_flag.paper2.bingo",  # Library microfiche reader flag paper 2 bingo text
    "tunic.flaghouse.entry.colorbook",  # Flag house entry color book text
    "tunic.wildlife.center.coffee",  # Wildlife coffee text
    "tunic.capitol_1.hall.boss.haveyougotit",  # Capitol 1 hall boss have you got it text
    "tunic.historicalsociety.basement.janitor",  # Basement janitor text
    "tunic.historicalsociety.collection_flag.gramps.recap",  # Collection flag gramps recap text
    "tunic.wildlife.center.wells.animals2",  # Wildlife wells animals 2 text
    "tunic.flaghouse.entry.flag_girl.symbol_recap",  # Flag house flag girl symbol recap text
    "tunic.historicalsociety.closet_dirty.photo",  # Dirty closet photo text
    "tunic.historicalsociety.stacks.outtolunch",  # Stacks out to lunch text
    "tunic.library.frontdesk.worker.wells_recap",  # Library worker wells recap text
    "tunic.historicalsociety.frontdesk.archivist_glasses.confrontation_recap",  # Archivist glasses confrontation recap text
    "tunic.capitol_0.hall.boss.talktogramps",  # Capitol 0 hall boss talk to gramps text
    "tunic.historicalsociety.closet.photo",  # Closet photo text
    "tunic.historicalsociety.collection.tunic",  # Collection tunic text
    "tunic.historicalsociety.closet.teddy.intro_0_cs_5",  # Closet teddy intro cutscene 5 text
    "tunic.historicalsociety.closet_dirty.gramps.archivist",  # Dirty closet gramps archivist text
    "tunic.historicalsociety.closet_dirty.door_block_talk",  # Dirty closet door block talk text
    "tunic.historicalsociety.entry.boss.flag_recap",  # Entry boss flag recap text
    "tunic.historicalsociety.frontdesk.archivist.need_glass_0",  # Archivist need glass 0 text
    "tunic.historicalsociety.entry.wells.talktogramps",  # Entry wells talk to gramps text
    "tunic.historicalsociety.frontdesk.block_magnify",  # Front desk block magnify text
    "tunic.historicalsociety.frontdesk.archivist.foundtheodora",  # Archivist found theodora text
    "tunic.historicalsociety.closet_dirty.gramps.nothing",  # Dirty closet gramps nothing text
    "tunic.historicalsociety.closet_dirty.door_block_clean",  # Dirty closet door block clean text
    "tunic.capitol_1.hall.boss.writeitup",  # Capitol 1 hall boss write it up text
    "tunic.library.frontdesk.worker.nelson_recap",  # Library worker nelson recap text
    "tunic.library.frontdesk.worker.hello_short",  # Library worker hello short text
    "tunic.historicalsociety.stacks.block",  # Stacks block text
    "tunic.historicalsociety.frontdesk.archivist.need_glass_1",  # Archivist need glass 1 text
    "tunic.historicalsociety.entry.boss.talktogramps",  # Entry boss talk to gramps text
    "tunic.historicalsociety.frontdesk.archivist.newspaper_recap",  # Archivist newspaper recap text
    "tunic.historicalsociety.entry.wells.flag_recap",  # Entry wells flag recap text
    "tunic.drycleaner.frontdesk.worker.done2",  # Dry cleaner worker done 2 text
    "tunic.library.frontdesk.worker.flag_recap",  # Library worker flag recap text
    "tunic.humanecology.frontdesk.block_0",  # Human ecology block 0 text
    "tunic.library.frontdesk.worker.preflag",  # Library worker pre flag text
    "tunic.historicalsociety.basement.gramps.seeyalater",  # Basement gramps see you later text
    "tunic.flaghouse.entry.flag_girl.hello_recap",  # Flag house flag girl hello recap text
    "tunic.historicalsociety.closet.doorblock",  # Closet door block text
    "tunic.drycleaner.frontdesk.worker.takealook",  # Dry cleaner worker take a look text
    "tunic.historicalsociety.basement.gramps.whatdo",  # Basement gramps what do text
    "tunic.library.frontdesk.worker.droppedbadge",  # Library worker dropped badge text
    "tunic.historicalsociety.entry.block_tomap2",  # Entry block to map 2 text
    "tunic.library.frontdesk.block_nelson",  # Library block nelson text
    "tunic.library.microfiche.block_0",  # Library microfiche block 0 text
    "tunic.historicalsociety.entry.block_tocollection",  # Entry block to collection text
    "tunic.historicalsociety.entry.block_tomap1",  # Entry block to map 1 text
    "tunic.historicalsociety.collection.gramps.look_0",  # Collection gramps look 0 text
    "tunic.library.frontdesk.block_badge",  # Library block badge text
    "tunic.historicalsociety.cage.need_glasses",  # Cage need glasses text
    "tunic.library.frontdesk.block_badge_2",  # Library block badge 2 text
    "tunic.kohlcenter.halloffame.block_0",  # Kohl center hall of fame block 0 text
    "tunic.capitol_0.hall.chap1_finale_c",  # Capitol 0 hall chapter 1 finale C text
    "tunic.capitol_1.hall.chap2_finale_c",  # Capitol 1 hall chapter 2 finale C text
    "tunic.capitol_2.hall.chap4_finale_c",  # Capitol 2 hall chapter 4 finale C text
    "tunic.wildlife.center.fox.concern",  # Wildlife fox concern text
    "tunic.drycleaner.frontdesk.block_0",  # Dry cleaner block 0 text
    "tunic.historicalsociety.entry.gramps.hub",  # Entry gramps hub text
    "tunic.humanecology.frontdesk.block_1",  # Human ecology block 1 text
    "tunic.drycleaner.frontdesk.block_1",  # Dry cleaner block 1 text
]

# Room lists for feature engineering
# Game locations that are analyzed for spatial behavior patterns
ROOM_LISTS = [
    "tunic.historicalsociety.entry",  # Historical society entry
    "tunic.wildlife.center",  # Wildlife center
    "tunic.historicalsociety.cage",  # Historical society cage
    "tunic.library.frontdesk",  # Library front desk
    "tunic.historicalsociety.frontdesk",  # Historical society front desk
    "tunic.historicalsociety.stacks",  # Historical society stacks
    "tunic.historicalsociety.closet_dirty",  # Historical society dirty closet
    "tunic.humanecology.frontdesk",  # Human ecology front desk
    "tunic.historicalsociety.basement",  # Historical society basement
    "tunic.kohlcenter.halloffame",  # Kohl center hall of fame
    "tunic.library.microfiche",  # Library microfiche
    "tunic.drycleaner.frontdesk",  # Dry cleaner front desk
    "tunic.historicalsociety.collection",  # Historical society collection
    "tunic.historicalsociety.closet",  # Historical society closet
    "tunic.flaghouse.entry",  # Flag house entry
    "tunic.historicalsociety.collection_flag",  # Historical society collection flag
    "tunic.capitol_1.hall",  # Capitol 1 hall
    "tunic.capitol_0.hall",  # Capitol 0 hall
    "tunic.capitol_2.hall",  # Capitol 2 hall
]

# Question categorization for different modeling approaches
# Questions are grouped based on their characteristics and required features

# Questions that use base features (common across all questions)
QUESTIONS_BASE_FEATURES = [2, 3, 9, 12, 13, 17, 18]

# Questions that require new feature engineering approaches
QUESTION_NEW_FEATURES = [4, 5, 8, 10, 11, 14, 15, 16]

# Questions that use new features with fold-based validation
QUESTION_NEW_FEATURES_FOLD = [1, 6, 7]

# Questions that use new features with random sampling
QUESTION_NEW_FEATURES_RANDOM = [8, 11, 14, 15]

# Questions that use 0.6 sampling rate
QUESTION_SAMPLE06 = [14, 15]

# Questions not used in training (test-only questions)
QUESTIONS_NOT_TRAIN = [2, 18]

# Training configuration parameters
N_SPLITS = 4  # Number of cross-validation folds
TRAIN = True  # Training mode flag

# Level group limits for temporal analysis
# Defines the boundaries for different game level groups
# Used to analyze student performance across different difficulty levels
LEVEL_LIMITS = {
    "0-4": (1, 4),  # Early game levels (beginner)
    "5-12": (4, 14),  # Middle game levels (intermediate)
    "13-22": (14, 19),  # Advanced game levels (expert)
}

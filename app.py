import asyncio
import copy
import dataclasses
import re
from collections import defaultdict
from functools import partial

import streamlit as st
from duckduckgo_search import DDGS

st.set_page_config(page_title="Fact Annotator", layout="wide", page_icon=":material/edit_note:")

import pandas as pd
import json, base64
from typing import List

from commons.datamodel import Entity, BiDirectionalRelation, merge_entities, disambiguate_entities, \
    disambiguate_relations, fuzzy_find_entity_by_alias
from image_processing.logo_preprocess import logo_to_url, NO_IMG_THUMB
from commons.website_parsing import get_organization_info
from widgets.entity_graph import draw_entity_graph
from integrations.wikipedia import entity_lookup_sync, verify_logo

from streamlit_theme import st_theme

# Third‑party component for cookies
from streamlit_cookies_manager import CookieManager

from integrations.gsheets_api import (
    extract_sheet_id,
    public_worksheets,
    worksheet_to_df,
    save_df_to_sheet, postprocess_df, prepare_df_to_export,
)

theme = (st_theme() or {}).get("base")

# ---------------------------------------------------------------------------
# 0. Persistent cookie‑backed session state ----------------------------------
# ---------------------------------------------------------------------------

COOKIE_KEY = "annot_state"

cookies = CookieManager()
if not cookies.ready():
    st.stop()

# Read cookie → prime st.session_state once per browser
cookie_val = cookies.get(COOKIE_KEY)
if cookie_val and "_cookie_init_done" not in st.session_state:
    try:
        saved_state = json.loads(base64.b64decode(cookie_val.encode()).decode())
        for k, v in saved_state.items():
            # Don’t overwrite if state was already set earlier in this run
            if k not in st.session_state:
                st.session_state[k] = v
    except Exception:
        pass
    st.session_state._cookie_init_done = True


# Initialise a counter used to bust worksheet‑name cache
if "ws_refresh_counter" not in st.session_state:
    st.session_state.ws_refresh_counter = 0

if "gs_synced" not in st.session_state:
    st.session_state.gs_synced = False

st.title(":material/edit_note: Fact Annotator")

data_selection_container = st.container(border=True)


gcp_creds = st.secrets.get("gcp_service_account")

# ---------------------------------------------------------------------------
# 1. Data‑source controls -----------------------------------------------------
sheet_url = None
with data_selection_container:
    if gcp_creds is None:
        st.warning(
            ":material/warning: Google Sheets integration is not available. "
            "Please refer to information at Fact Annotator GitHub page for more information."
        )
    else:
        sheet_url = st.text_input(
            "Google Sheets URL (optional)",
            value=st.session_state.get("gs_url", ""),
            placeholder="https://docs.google.com/spreadsheets/d/…",
            key="gs_url",
            help=f"Make the spreadsheet public or add {gcp_creds.get('client_email')} as Editor"
        )

auto_sheet_id = extract_sheet_id(sheet_url) if sheet_url else None

# Warn if no URL – state will be lost
use_gs = False
with data_selection_container:
    if not sheet_url:
        st.warning(
            ":material/warning: You didn’t provide a Google Sheets URL – changes are kept only in the browser session "
            "and will be lost on refresh."
        )
    else:
        use_gs = st.checkbox(
            "Use Google Sheets as data source (instead of uploaded CSV)",
            value=st.session_state.get("use_gs", False),
            key="use_gs",
        )

# Option to load from Google Sheets instead of CSV

uploaded_file = None
if not use_gs:
    with data_selection_container:
        uploaded_file = st.file_uploader(
            "Upload a CSV file", type=["csv"], accept_multiple_files=False
        )

gs_salt = auto_sheet_id if auto_sheet_id is not None else ""
source_salt = str(uploaded_file) if uploaded_file is not None and not use_gs else gs_salt

# ---------------------------------------------------------------------------
# 2. Worksheet selector (if URL present) -------------------------------------


@st.cache_data(show_spinner=False)
def cached_public_worksheets(sheet_id: str, _refresh_token: int = 0) -> List[str]:
    """Cached wrapper around public_worksheets to avoid repeated HTTP calls."""
    return public_worksheets(sheet_id)


def filter_json_columns(col: str) -> bool:
    return col.endswith('_entity') or col.endswith('_relation')


worksheet_names: List[str] = []
selected_ws: str | None = None

if auto_sheet_id:
    with data_selection_container:
        @st.fragment()
        def gs_update():
            global selected_ws

            # layout: selector + refresh button side‑by‑side
            refresh_col, ws_col, save_col = st.columns([1, 11, 11], vertical_alignment="bottom")
            info_container = st.container()

            def _update_counter():
                if "ws_refresh_counter" not in st.session_state:
                    st.session_state.ws_refresh_counter = 0
                st.session_state.ws_refresh_counter += 1

            with refresh_col:
                st.button(":material/cached:", help="Refresh worksheet list", on_click=_update_counter)

            worksheet_names = (
                cached_public_worksheets(auto_sheet_id, st.session_state.ws_refresh_counter)
                or ["Sheet1"]
            )

            with ws_col:
                selected_ws = st.selectbox(
                    "Worksheet to load / save",
                    worksheet_names,
                    key="worksheet_select" + gs_salt,
                )

            global source_salt
            source_salt += selected_ws if selected_ws and use_gs else ""

            with save_col:
                def _save_df():
                    with info_container:
                        if not sheet_url:
                            st.error("Please provide a Google Sheets URL first.")
                        elif not selected_ws:
                            st.error("Please choose a worksheet to save to.")
                        else:
                            save_df_to_sheet(sheet_url, selected_ws, prepare_df_to_export(
                                st.session_state.df, filter_json_columns
                            ))
                            # New worksheet may have been created → refresh list next run
                            st.session_state.ws_refresh_counter += 1
                            st.session_state.gs_synced = True

                st.button(
                    f":material/save: Save to {selected_ws}",
                    type="primary",
                    use_container_width=True,
                    on_click=_save_df
                )

            with info_container:
                if st.session_state.gs_synced:
                    st.info(":material/data_check: Data is synced with Google Sheets.")
                else:
                    st.warning(":material/data_alert: Data is not synced with Google Sheets. Please save.")

        gs_update()

# ---------------------------------------------------------------------------
# 3. Load dataframe ----------------------------------------------------------

def init_dataframe() -> pd.DataFrame | None:
    """Return a DataFrame from chosen source or None if unavailable."""
    if use_gs and auto_sheet_id and selected_ws:
        try:
            st.session_state.gs_synced = True
            return postprocess_df(worksheet_to_df(auto_sheet_id, selected_ws), filter_json_columns)
        except Exception as err:
            st.error(f"Could not load Google Sheet: {err}")
            return None
    elif uploaded_file is not None:
        st.session_state.gs_synced = False
        return postprocess_df(pd.read_csv(uploaded_file), filter_json_columns)
    return None

# Generate a signature so we know when to reload
source_signature = (
    f"gs:{auto_sheet_id}:{selected_ws}:{use_gs}"
    if use_gs
    else uploaded_file.name + str(uploaded_file.file_id)
    if uploaded_file
    else "none"
)

if (
    "df" not in st.session_state
    or st.session_state.get("_source_sig") != source_signature
):
    df_loaded = init_dataframe()
    if df_loaded is not None:
        st.session_state.df = df_loaded
        st.session_state._source_sig = source_signature

# ---------------------------------------------------------------------------
# 4. Annotation UI -----------------------------------------------------------


def get_entity_types(columns: list[str]) -> list[str]:
    result = []
    for column in columns:
        if column.endswith("_entity"):
            result.append(column.split("_")[0])
    return result


def get_relation_types(columns: list[str]) -> list[str]:
    result = []
    for column in columns:
        if column.endswith("_relation"):
            result.append(column.split("_")[0])
    return result


def highlight_matches(
    text: str,
    color2aliases: dict[str, list[str]],
    *,
    case_sensitive: bool = True,
) -> str:
    """
    Wrap every alias that appears in *text* with <u> … </u> tags.

    Parameters
    ----------
    text : str
        The original text.
    color2aliases : dict[str, list[str]]
        Words/phrases to underline if they occur in *text*.
    case_sensitive : bool, default False
        If False, matching is case-insensitive.

    Returns
    -------
    str
        The text with matching substrings wrapped in <u> tags.
        Safe to render via st.markdown(..., unsafe_allow_html=True).
    """
    alias2color = {
        alias: color
        for color, aliases in color2aliases.items()
        for alias in aliases
    }
    # 1) keep only aliases that are actually present (fast short-circuit)
    detected = [a for a in alias2color.keys() if a and (a in text if case_sensitive else a.lower() in text.lower())]
    if not detected:
        return text

    # 2) longest first prevents partial-match overlaps
    detected = sorted(set(detected), reverse=True)

    # 3) compile one regex that matches ANY alias
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile("|".join(map(re.escape, detected)), flags=flags)

    # 4) substitute with underline markup
    return pattern.sub(lambda m: f":{alias2color[m.group(0)]}-badge[{m.group(0)}]", text)


if "df" in st.session_state:
    @st.fragment
    def updatable_elements():
        df = st.session_state.df

        data_view_container = st.container(border=True)
        data_modification_container = st.container(border=True)
        annotation_container = st.container(border=True)

        if "Completed" not in df.columns:
            df["Completed"] = False
        else:
            df["Completed"] = df["Completed"].astype(bool)

        entity_types = get_entity_types(df.columns)
        if "entity_types" not in st.session_state or st.session_state.entity_types is None:
            st.session_state.entity_types = entity_types
        else:
            for selected_entity_type in st.session_state.entity_types:
                column_name = f"{selected_entity_type}_entity"
                if column_name not in df.columns:
                    df[column_name] = [[] for _ in range(len(df))]
        st.session_state.entity_types = sorted(set(st.session_state.entity_types))

        relation_types = get_relation_types(df.columns)
        if "relation_types" not in st.session_state or st.session_state.relation_types is None:
            st.session_state.relation_types = relation_types
        else:
            for relation_type in st.session_state.relation_types:
                column_name = f"{relation_type}_relation"
                if column_name not in df.columns:
                    df[column_name] = [[] for _ in range(len(df))]
        st.session_state.relation_types = sorted(set(st.session_state.relation_types))

        global_ent_dicts = []
        for ent_type in st.session_state.entity_types:
            column_name = f"{ent_type}_entity"
            # turn each list into separate rows, drop the NaNs that come from
            # empty lists, and collect everything in one go
            global_ent_dicts.extend(
                df[column_name].explode().dropna().tolist()
            )

        global_entities = disambiguate_entities([Entity(**ent_dict) for ent_dict in global_ent_dicts])
        id2global_entities = defaultdict(list)
        for ent in global_entities:
            id2global_entities[ent.id].append(ent)

        alias2global_entities = defaultdict(list)
        for ent in global_entities:
            for alias in ent.aliases:
                alias2global_entities[alias].append(ent)

        global_rel_dicts = []
        for rel in st.session_state.relation_types:
            column_name = f"{rel}_relation"
            # turn each list into separate rows, drop the NaNs that come from
            # empty lists, and collect everything in one go
            global_rel_dicts.extend(
                df[column_name].explode().dropna().tolist()
            )

        global_relations = disambiguate_relations([BiDirectionalRelation(**rel_dict) for rel_dict in global_rel_dicts])
        id_pair2global_relations = defaultdict(int)
        for rel in global_relations:
            id_pair = (rel.from_id, rel.to_id)
            id_pair2global_relations[id_pair] += 1

        with data_view_container:
            exp_data, exp_entities, exp_relations = st.columns(3, gap="large")

            def _export_data():
                non_json_columns = [col for col in df.columns if not filter_json_columns(col)]
                data_df = df[non_json_columns]

                @st.dialog(f"Data table")
                def preview_data():
                    st.dataframe(data_df)

                preview_data()

            with exp_data:
                st.button("Download data", on_click=_export_data, use_container_width=True)

            def _export_entities():
                data = []
                for ent_type in st.session_state.entity_types:
                    for row_idx in range(len(df)):
                        column_name = f"{ent_type}_entity"

                        for ent_dict in df.at[row_idx, column_name]:
                            datapoint = copy.deepcopy(ent_dict)
                            datapoint["row_idx"] = row_idx
                            datapoint["column"] = "?"  # FIXME: we need to keep track of column in the entity
                            data.append(datapoint)

                data_df = pd.DataFrame(data)

                @st.dialog(f"Entities table")
                def preview_relations():
                    st.dataframe(data_df)

                preview_relations()

            with exp_entities:
                st.button("Download entities", on_click=_export_entities, use_container_width=True)

            def _export_relations():
                data = []
                for rel_type in st.session_state.relation_types:
                    for row_idx in range(len(df)):
                        column_name = f"{rel_type}_relation"

                        for rel_dict in df.at[row_idx, column_name]:
                            datapoint = copy.deepcopy(rel_dict)
                            datapoint["row_idx"] = row_idx
                            datapoint["column"] = "?"  # FIXME: we need to keep track of column in the entity
                            data.append(datapoint)

                data_df = pd.DataFrame(data)

                @st.dialog(f"Relations table")
                def preview_relations():
                    st.dataframe(data_df)

                preview_relations()

            with exp_relations:
                st.button("Download relations", on_click=_export_relations, use_container_width=True)

        def visualize_new_entity(
                new_ent: Entity,
                *,
                thumb_column=None,
                description_column=None,
                omit_description: bool = False,
        ):
            old_entity = merge_entities(id2global_entities.get(new_ent.id, []))

            is_new = False
            new_aliases = set(new_ent.aliases)
            new_ent_copy = copy.deepcopy(new_ent)
            if old_entity is None:
                is_new = True
            else:
                new_aliases = set(new_ent.aliases).difference(set(old_entity.aliases))
                new_ent_copy.merge(old_entity)

            if thumb_column is None or description_column is None:
                thumb_column, description_column = st.columns([1, 2])

            with thumb_column:
                st.image(logo_to_url(new_ent_copy.thumbnail or NO_IMG_THUMB, theme=theme))

            with description_column:
                label = "**" + new_ent_copy.name + "**"
                description = new_ent_copy.description or "Not available"
                url = new_ent_copy.url  # may be None / ""

                aliases = [
                    alias + (' :blue-badge[New]' if alias in new_aliases else "")
                    for alias in new_ent_copy.aliases
                ]

                suffix = " :blue-badge[New]" if is_new else ""
                if not omit_description:
                    suffix += "  \n**Description**: " + description
                suffix += "  \n**Aliases**: " + ", ".join(aliases)

                if url:
                    st.markdown(f"[{label}]({url})" + suffix)
                else:
                    st.markdown(label + suffix)

        global source_salt
        with data_view_container:
            show_columns = [col for col in df.columns if not filter_json_columns(col)]
            columns_sep = "<|COLUMNS|>:"
            if columns_sep not in source_salt:
                source_salt += columns_sep + ','.join(show_columns) + str(len(df))
            with st.expander("Data preview", expanded=True):
                st.dataframe(df[show_columns], height=200)

        with data_view_container:
            with st.expander("Types settings"):
                entity_types_columns = st.columns([2, 1])
                with entity_types_columns[0]:
                    st.multiselect("Entity types", entity_types, default=entity_types)

                with entity_types_columns[1]:
                    entity_add_columns = st.columns([5, 1], gap="small", vertical_alignment="bottom")
                    with entity_add_columns[0]:
                        st.text_input("New entity type", key="new_entity_type_name")

                    new_type = st.session_state.new_entity_type_name
                    disabled_ent_type = (not new_type) or new_type in st.session_state.entity_types

                    def _submit_entity_type():
                        st.session_state.entity_types.append(st.session_state.new_entity_type_name)
                        st.session_state.new_entity_type_name = ''

                    with entity_add_columns[1]:
                        st.button(
                            ":material/add:",
                            key="add_entity_type",
                            type="primary",
                            on_click=_submit_entity_type,
                            disabled=disabled_ent_type,
                        )

                relation_types_columns = st.columns([2, 1])
                with relation_types_columns[0]:
                    st.multiselect("Relation types", relation_types, default=relation_types)

                with relation_types_columns[1]:
                    relation_add_columns = st.columns([5, 1], gap="small", vertical_alignment="bottom")
                    with relation_add_columns[0]:
                        st.text_input("New relation type", key="new_relation_type_name")

                    new_type = st.session_state.new_relation_type_name
                    disabled_rel_type = (not new_type) or new_type in st.session_state.relation_types

                    def _submit_relation_type():
                        st.session_state.relation_types.append(st.session_state.new_relation_type_name)
                        st.session_state.new_relation_type_name = ''

                    with relation_add_columns[1]:
                        st.button(
                            ":material/add:",
                            key="add_relation_type",
                            type="primary",
                            on_click=_submit_relation_type,
                            disabled=disabled_rel_type
                        )

        with data_modification_container:
            col_entity_picker, col_md = st.columns([1, 3], gap="small")

        with col_md:
            col_field, col_index, col_compl = st.columns([1, 1, 1], gap="large", vertical_alignment="bottom")
            with col_field:
                field_selector_key = "field_selector" + source_salt
                annotatable_columns = [
                    c for c in df.columns
                    if c != "Completed" and not filter_json_columns(c) and df.loc[:, c].dtype in ["O", str]
                ]
                selected_field = st.selectbox(
                    "Field (column) to annotate",
                    annotatable_columns,
                    key=field_selector_key,
                )

            with col_index:
                row_selector_key = "row_selector" + source_salt
                first_open_row = (
                    int(df.index[df["Completed"] == False][0])
                    if (df["Completed"] == False).any()
                    else 0
                )
                if row_selector_key not in st.session_state:
                    st.session_state[row_selector_key] = first_open_row

                max_row = len(df) - 1
                selected_row = st.number_input(
                    "Row index",
                    min_value=0,
                    max_value=max_row,
                    step=1,
                    key=row_selector_key,
                    format="%d",
                )
                selected_row = int(selected_row)
            with col_compl:
                completed_key = f"completed_checkbox_{selected_row}" + source_salt

                def switch_df_value():
                    df.at[selected_row, "Completed"] = ~df.at[selected_row, "Completed"]
                    st.session_state.gs_synced = False

                is_row_completed = st.checkbox(
                    "Completed",
                    key=completed_key,
                    value=bool(df.at[selected_row, "Completed"]),
                    on_change=switch_df_value,
                )

        local_ent_dicts = []
        for ent_type in st.session_state.entity_types:
            column_name = f"{ent_type}_entity"
            local_ent_dicts.extend(df.at[st.session_state[row_selector_key], column_name])

        salt = f'_{selected_field}_{selected_row}'

        local_entities = disambiguate_entities([Entity(**ent_dict) for ent_dict in local_ent_dicts])
        # store disambiguated entities in the df
        for ent_type in st.session_state.entity_types:
            column_name = f"{ent_type}_entity"
            filtered_entities = list(filter(lambda e: e.type == ent_type, local_entities))
            current_entities = [Entity(**ent_dict) for ent_dict in df.at[selected_row, column_name]]
            if tuple(sorted(filtered_entities)) != tuple(sorted(current_entities)):
                df.at[selected_row, column_name] = [
                    dataclasses.asdict(ent) for ent in filtered_entities
                ]
                st.session_state.gs_synced = False

        id2local_entities = defaultdict(list)
        for ent in local_entities:
            id2local_entities[ent.id].append(ent)

        alias2local_entities = defaultdict(list)
        for ent in local_entities:
            for alias in ent.aliases:
                alias2local_entities[alias].append(ent)

        local_rel_dicts = []
        for rel in st.session_state.relation_types:
            column_name = f"{rel}_relation"
            local_rel_dicts.extend(df.at[selected_row, column_name])

        local_relations = disambiguate_relations([BiDirectionalRelation(**rel_dict) for rel_dict in local_rel_dicts])

        # store disambiguated relations in the df
        for rel_type in st.session_state.relation_types:
            column_name = f"{rel_type}_relation"
            filtered_relations = list(filter(lambda r: r.type == rel_type, local_relations))
            current_relations = [BiDirectionalRelation(**rel_dict) for rel_dict in df.at[selected_row, column_name]]
            if True or tuple(sorted(filtered_relations)) != tuple(sorted(current_relations)):
                df.at[selected_row, column_name] = [
                    dataclasses.asdict(rel) for rel in filtered_relations
                ]
                st.session_state.gs_synced = False

        id_pair2local_relations = defaultdict(list)
        for rel in local_relations:
            id_pair = (rel.from_id, rel.to_id)
            id_pair2local_relations[id_pair].append(rel)

        with data_view_container:
            toggle_graph, empty_space, toggle_scope = st.columns([1, 3, 1])

            with toggle_graph:
                visualize_graph = st.toggle("Graph visualization", value=False)

            if visualize_graph:
                with toggle_scope:
                    graph_global_scope = st.toggle("Global scope", value=False)

                target_entities = global_entities if graph_global_scope else local_entities
                target_relations = global_relations if graph_global_scope else local_relations

                with st.container(border=True, height=540):
                    draw_entity_graph(target_entities, target_relations, theme=theme)

        text = ''
        if selected_field:
            text = df.at[selected_row, selected_field]

        with col_entity_picker:
            st.subheader("Add alias")
            col_type, col_mode = st.columns([2, 1], vertical_alignment="top")
            with col_type:
                selected_entity_type = st.selectbox(
                    "Entity type",
                    key="entity_type_selection" + source_salt,
                    options=st.session_state.entity_types
                )

            with col_mode:
                format_map = {
                    "Search": ":material/language:",
                    "Edit": ":material/edit:"
                }
                add_alias_mode = st.pills(
                    "Add mode",
                    ["Search", "Edit"],
                    selection_mode="single",
                    help=f"Choose {format_map['Search']} to look up new entity on Wikipedia or {format_map['Edit']} "
                         "to add alias to already annotated entity.",
                    format_func=lambda k: format_map[k],
                    default="Search"
                )

            column_name = f"{selected_entity_type}_entity"

            col_edit, col_add = st.columns([4, 1], vertical_alignment="bottom")
            with col_edit:
                alias = st.text_input("Alias", key="alias_edit" + source_salt + salt)

            def add_entity(e: Entity):
                df.at[selected_row, column_name].append(dataclasses.asdict(e))
                st.session_state.gs_synced = False

            def new_entity_widget(search_term):
                with st.container(border=True):
                    entity_column, select_column = st.columns([5, 1])

                with select_column:
                    if "ddgs_lookup" not in st.session_state:
                        st.session_state.ddgs_lookup = None
                    ddgs_lookup = st.pills(
                        "Find",
                        [True],
                        key="ddgs_lookup",
                        help="Look up best match on DuckDuckGoSearch",
                        format_func=lambda _: ":material/search:",
                    )

                if ddgs_lookup:
                    search_engine = DDGS()

                    ddgs_result = search_engine.text(keywords=search_term, max_results=1)[0]
                    url = ddgs_result.get("href")
                    description = ddgs_result.get("body")

                    ddgs_result = search_engine.images(keywords=search_term, max_results=1)[0]
                    image = ddgs_result.get("image")

                    if not description or not image:
                        info = asyncio.run(get_organization_info(url))

                        if description is None:
                            description = info.get("description")

                        if image is None:
                            image = info.get("image")

                    entity_new = Entity(
                        name=alias,
                        type=selected_entity_type,
                        id=url,
                        url=url,
                        thumbnail=image if verify_logo(image) else None,
                        description=description,
                    )
                else:
                    entity_new = Entity(name=alias, type=selected_entity_type, thumbnail=NO_IMG_THUMB)

                with entity_column:
                    visualize_new_entity(entity_new)

                with select_column:
                    if st.button(":material/add:", key="create_new", on_click=partial(add_entity, e=entity_new)):
                        st.rerun()

            def entity_option_widget(e: Entity, option_idx: int):
                with st.container(border=True):
                    entity_column, select_column = st.columns([6, 1])

                with entity_column:
                    visualize_new_entity(e)

                with select_column:
                    if st.button(
                            ":material/add:",
                            key=f"{option_idx}_submit",
                            type="primary",
                            on_click=partial(add_entity, e=e)
                    ):
                        st.rerun()

            def _edit_entity():
                @st.dialog(f"Choose entity for {alias} ({selected_entity_type})")
                def choose_entity():
                    search_term = st.text_input(":material/edit: Entities search term:", value=alias)
                    options: list[Entity] = fuzzy_find_entity_by_alias(search_term, global_entities)

                    if not len(options):
                        st.warning("Could not find any matching entities")
                    else:
                        with st.container(height=300):
                            for option_idx, option in enumerate(options):
                                option_with_new_alias = copy.deepcopy(option)
                                option_with_new_alias.aliases.append(alias)

                                entity_option_widget(option_with_new_alias, option_idx)

                choose_entity()

            def _look_up_entity():
                @st.dialog(f"Choose entity for {alias} ({selected_entity_type})")
                def choose_wiki_entity():
                    search_term = st.text_input(":material/language: Wikipedia search term:", value=alias)
                    options = entity_lookup_sync(search_term)

                    if not len(options):
                        st.warning("Could not find any Wikipedia entities")
                    else:
                        with st.container(height=300):
                            for option_idx, option in enumerate(options):
                                entity_opt = Entity(
                                    name=option.get("name", alias),
                                    type=selected_entity_type,
                                    id=option.get("id"),
                                    description=option.get("description"),
                                    url=option.get("url"),
                                    thumbnail=option.get("thumbnail", NO_IMG_THUMB),
                                    aliases=[alias]
                                )

                                entity_option_widget(entity_opt, option_idx)

                    new_entity_widget(search_term)

                choose_wiki_entity()

            add_action = _look_up_entity if add_alias_mode == "Search" else _edit_entity

            alias_ready = True
            if len(alias) < 2:
                st.warning(":material/warning: Alias is too short")
                alias_ready = False
            elif add_alias_mode is None:
                st.warning(":material/warning: Select add mode")
                alias_ready = False
            elif selected_entity_type is None:
                st.warning(":material/warning: Configure entity type")
                alias_ready = False
            elif alias not in text:
                st.warning(":material/warning: Alias is not mentioned in text")
                alias_ready = False
            elif alias in alias2global_entities:
                st.warning(":material/warning: Alias already exists")
                alias_ready = False
            else:
                st.info(":material/check: Alias is good")

            with col_add:
                st.button(
                    ":material/add:",
                    on_click=add_action,
                    type="primary",
                    disabled=not alias_ready
                )

            potential_matches: list[Entity] = []
            for existing_alias in alias2global_entities.keys():
                # if alias is in local, then it has been already added
                if existing_alias not in alias2local_entities and existing_alias in text:
                    potential_matches.extend(filter(
                        lambda e: e.type == selected_entity_type,
                        alias2global_entities[existing_alias]
                    ))
            potential_matches = disambiguate_entities(potential_matches)

            st.subheader("Alias matches:")

            with st.container(height=400, border=True):
                for idx, match in enumerate(potential_matches):
                    with st.container(border=True):
                        thumb_col, desc_col = st.columns([1, 3])

                    visualize_new_entity(
                        match,
                        thumb_column=thumb_col,
                        description_column=desc_col,
                        omit_description=True
                    )

                    def add_entity(e: Entity):
                        df.at[selected_row, column_name].append(dataclasses.asdict(e))
                        st.session_state.gs_synced = False

                    with thumb_col:
                        st.button(
                            ":material/add:",
                            type="primary",
                            key=f"add_match_{idx}",
                            on_click=partial(add_entity, e=match)
                        )

            with col_md:
                if selected_field:
                    color_code = {'red': alias2local_entities.keys()}
                    if alias_ready:
                        color_code['blue'] = [alias]

                    text_md = highlight_matches(
                        text.replace("$", "\\$").replace("`", "'"),
                        color_code
                    )

                    with st.container(height=678, border=True):
                        st.markdown(text_md, unsafe_allow_html=True)  # allow html for underline

        with data_view_container:
            st.subheader("Modify relations")
            act_type, controls = st.columns([3, 3 + 8 + 8 + 1], vertical_alignment="top")
            with controls:
                rel_type, from_ent, to_ent, btn = st.columns([3, 8, 8, 1], vertical_alignment="bottom")

            with rel_type:
                relation_type = st.selectbox("Relation type", st.session_state.relation_types)

            with act_type:
                action_type = st.pills("Action type", ["Add", "Delete"], default="Add")
                if action_type is not None:
                    st.session_state.last_action_type = action_type

            with from_ent:
                from_selection = sorted(local_entities, key=lambda e: e.name)

                if ("relation_from_entity" + salt) not in st.session_state:
                    value = from_selection[0] if len(from_selection) > 0 else None
                    st.session_state["relation_from_entity" + salt] = value

                from_entity = st.selectbox(
                    "From entity",
                    from_selection,
                    key="relation_from_entity" + source_salt + salt
                )
            with to_ent:
                if from_entity is not None:
                    current_to_ids = [rel.to_id for rel in local_relations if rel.from_id == from_entity.id]
                    current_from_ids = [rel.from_id for rel in local_relations if rel.to_id == from_entity.id]

                    # FIXME: this assumes all relations are bi-directional
                    restricted_set = set(current_to_ids).union(current_from_ids)

                    def is_valid(ent_id):
                        if action_type == "Add":
                            return ent_id != from_entity.id and ent_id not in restricted_set
                        else:
                            return ent_id != from_entity.id and ent_id in restricted_set

                    valid_targets = sorted([ent for ent in local_entities if is_valid(ent.id)], key=lambda e: e.name)
                    if ("relation_to_entity" + salt) in st.session_state \
                            and st.session_state["relation_to_entity" + salt] not in valid_targets:
                        st.session_state["relation_to_entity" + salt] = valid_targets[0] if len(valid_targets) else None
                else:
                    valid_targets = []

                to_entity = st.selectbox(
                    "To entity",
                    valid_targets,
                    key="relation_to_entity" + source_salt + salt,
                    disabled=from_entity is None
                )

            def _submit_relation():
                column_name = f"{relation_type}_relation"
                df.at[selected_row, column_name].append(dataclasses.asdict(BiDirectionalRelation(
                    from_id=from_entity.id,
                    to_id=to_entity.id,
                    type=relation_type,
                )))
                st.session_state.gs_synced = False

            def _delete_relation():
                column_name = f"{relation_type}_relation"
                before = df.at[selected_row, column_name]
                after = filter(lambda r: r.from_id != from_entity.id or r.to_id != to_entity.id, before)
                df.at[selected_row, column_name].clear()
                df.at[selected_row, column_name].extend(after)
                st.session_state.gs_synced = False

            action = _submit_relation if action_type == "Add" else _delete_relation
            with btn:
                st.button(
                    ":material/add:" if st.session_state.last_action_type == "Add" else ":material/delete:",
                    key="add_relation",
                    type="primary",
                    on_click=action,
                    disabled=action_type is None or from_entity is None or to_entity is None,
                )

        with annotation_container:
            with st.expander("Annotations list"):
                kind_col, type_col, scope_col = st.columns([2, 6, 2], vertical_alignment="bottom")
                with kind_col:
                    list_kinds = st.pills(
                        "Kind",
                        ["Entity", "Relation"],
                        default=["Entity", "Relation"],
                        selection_mode='multi'
                    )

                target_types = []
                if "Entity" in list_kinds:
                    target_types.extend(st.session_state.entity_types)

                if "Relation" in list_kinds:
                    target_types.extend(st.session_state.relation_types)

                with type_col:
                    list_types = st.pills("Types", target_types, default=target_types, selection_mode='multi')

                with scope_col:
                    global_scope = st.toggle("Global scope", key="global_scope_list")

                target_entities = global_entities if global_scope else local_entities
                target_entities = filter(lambda e: e.type in list_types, target_entities)

                with st.container(border=True, height=800):
                    for idx, ent in enumerate(target_entities):
                        with st.container(border=True):
                            thumb_col, desc_col, action_col = st.columns([3, 15, 1])
                            visualize_new_entity(ent, thumb_column=thumb_col, description_column=desc_col)

                            with action_col:
                                def delete_entity(e: Entity):
                                    st.session_state.gs_synced = False

                                    target_column = f"{ent.type}_entity"
                                    before = df.at[selected_row, target_column]
                                    after = list(filter(lambda d: d["id"] != ent.id, before))
                                    df.at[selected_row, column_name].clear()
                                    df.at[selected_row, column_name].extend(after)

                                    for rel_type in st.session_state.relation_types:
                                        target_column = f"{rel_type}_relation"
                                        before = df.at[selected_row, target_column]
                                        after = list(filter(
                                            lambda d: d["to_id"] != ent.id and d["from_id"] != ent.id,
                                            before
                                        ))
                                        df.at[selected_row, target_column].clear()
                                        df.at[selected_row, target_column].extend(after)

                                if not global_scope:
                                    st.button(
                                        ":material/delete:",
                                        help="Delete entity and associated relations",
                                        key=f"delete_{idx}",
                                        type="primary",
                                        on_click=partial(delete_entity, e=ent)
                                    )

    updatable_elements()

else:
    st.info(
        ":material/arrow_upward: Upload a CSV file **or** enter a Google Sheets URL "
        "and check 'Use Google Sheets as data source' to begin."
    )

# ---------------------------------------------------------------------------
# 5. Persist relevant session_state into a cookie ----------------------------

_state_snapshot = {
    "gs_url": st.session_state.get("gs_url"),
    "use_gs": st.session_state.get("use_gs"),
    "worksheet_select": st.session_state.get("worksheet_select" + gs_salt),
}
try:
    cookies[COOKIE_KEY] = base64.b64encode(json.dumps(_state_snapshot).encode()).decode()
    cookies.save()
except Exception:
    pass

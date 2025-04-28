from commons.datamodel import Entity, BiDirectionalRelation
from streamlit_agraph import agraph, Node, Edge, Config

from image_processing.logo_preprocess import logo_to_url

import streamlit as st


def draw_entity_graph(entities: list[Entity], relations: list[BiDirectionalRelation], *, theme=None) -> None:
    config = Config(
        width="100%",
        height=500,
        levelSeparation=100,
        nodeSpacing=100,
        directed=False,
        physics=True,
        hierarchical=False,
        node={'labelProperty': 'label'},
        maxZoom=2,
        minZoom=0.5,
        staticGraphWithDragAndDrop=False,
        staticGraph=False,
        interaction={
            "zoomView": False,
            "dragView": False
        }
        # **kwargs
    )
    config.physics["repulsion"] = {
        "nodeDistance": 10,
        "centralGravity": 1.0
    }
    config.physics["adaptiveTimestep"] = False

    nodes = []
    for entity in entities:
        if entity.thumbnail is not None:
            nodes.append(Node(
                id=entity.id,
                title=f"Description: {entity.description}\n"
                      f"Aliases: {', '.join(entity.aliases)}\n"
                      f"URL: {entity.url}",
                image=logo_to_url(entity.thumbnail, theme=theme),
                shape="image",
                size=25,
            ))
        else:
            nodes.append(Node(
                label=entity.name,
                id=entity.id,
                title=f"Description: {entity.description}\n"
                      f"Aliases: {', '.join(entity.aliases)}\n"
                      f"URL: {entity.url}",
                shape="text",
                font={
                    "size": 25,
                    "color": "#000000" if theme != "dark" else "#FFFFFF",
                },
                size=50,
            ))

    edge_color = st.get_option("theme.primaryColor") or "#FF4B4B"

    edges = []
    for relation in relations:
        edges.append(Edge(
            source=relation.from_id,
            target=relation.to_id,
            color=edge_color,
            width=5,
            smooth={
                "enabled": True,
            }
        ))

    return agraph(
        nodes=nodes,
        edges=edges,
        config=config
    )

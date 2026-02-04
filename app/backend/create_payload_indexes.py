#!/usr/bin/env python3
"""
Create payload indexes for efficient filtering on 600K+ recipes.

RUN THIS SCRIPT when:
1. Searches are timing out due to numerical filters
2. After indexing a significant portion of recipes (e.g., 200K+)

This creates indexes for:
- Numerical fields (Z_BRIX, Z_PH, Z_FRUCHTG, etc.) - for range queries
- Version/country fields - for exact match filtering
"""

import os
import sys
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType, TextIndexParams, TokenizerType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All numerical fields that need FLOAT indexes for range queries
NUMERICAL_FIELDS = [
    'Z_BRIX', 'Z_PH', 'ZM_PH', 
    'Z_VISK20S', 'Z_VISK20S_7C', 'Z_VISK30S', 'Z_VISK60S', 'Z_VISKHAAKE',
    'Z_VISK4S', 'Z_VISK70S',  # Additional viscosity fields
    'ZMX_DD103', 'ZMX_DD102',
    'ZM_AW', 'Z_FGAW',
    'Z_FRUCHTG', 'ZMX_DD108', 'Z_AW',
    'ZMX_DD109', 'Z_DOSIER',
    'Z_ZUCKER', 'Z_FETTST', 'Z_PROT', 'Z_SALZ',
    'Z_VERHAN',  # Dilution ratio
    'Z_TRMA',    # Dry matter %
    'Z_PAST',    # Pasteurization temperature
    'Z_HALTB',   # Shelf life (months)
]

# All 60 specified fields for keyword indexes
SPECIFIED_FIELDS = [
    'Z_MAKTX', 'Z_INH01', 'Z_WEIM', 'Z_KUNPROGRU', 'Z_PRODK',
    'Z_INH07', 'Z_KOCHART', 'Z_KNOGM', 'Z_INH08', 'Z_INH10', 'Z_INH12',  # Added Z_INH10 (GMO)
    'ZMX_TIPOALERG', 'Z_INH02', 'Z_INH03', 'Z_INH19', 'Z_INH04',
    'Z_INH18', 'Z_INH05', 'Z_INH09', 'Z_INH06', 'Z_INH06Z',
    'Z_FSTAT', 'Z_INH21', 'Z_INH13', 'Z_INH14', 'Z_INH15',
    'Z_INH16', 'Z_INH20', 'Z_STABGU', 'Z_STABCAR', 'Z_STAGEL',
    'Z_STANO', 'Z_INH17', 'Z_BRIX', 'Z_PH', 'ZM_PH',
    'Z_VISK20S', 'Z_VISK20S_7C', 'Z_VISK30S', 'Z_VISK60S', 'Z_VISKHAAKE',
    'Z_VISK4S', 'Z_VISK70S',  # Additional viscosity fields
    'ZMX_DD103', 'ZMX_DD102', 'ZM_AW', 'Z_FGAW', 'Z_FRUCHTG',
    'ZMX_DD108', 'Z_AW', 'Z_FLST', 'Z_PP', 'ZMX_DD109',
    'Z_DOSIER', 'Z_ZUCKER', 'Z_FETTST', 'ZMX_DD104', 'Z_PROT',
    'Z_SALZ', 'Z_INH01K', 'Z_INH01H', 'Z_DAIRY', 'Z_BFS',
    'Z_VERHAN', 'Z_TRMA', 'Z_PAST', 'Z_HALTB',  # Additional fields
]


def create_payload_indexes(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "food_recipes_two_step"
):
    """Create all necessary payload indexes for fast filtering."""
    
    print("=" * 70)
    print("CREATING PAYLOAD INDEXES FOR QDRANT")
    print(f"Host: {qdrant_host}:{qdrant_port}")
    print(f"Collection: {collection_name}")
    print("=" * 70)
    
    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=300)
    
    # Check collection exists
    try:
        collection_info = client.get_collection(collection_name)
        print(f"\n✓ Collection found: {collection_info.points_count:,} points")
    except Exception as e:
        print(f"\n✗ Collection not found: {e}")
        return False
    
    created_count = 0
    skipped_count = 0
    
    # 1. Create FLOAT indexes for numerical fields (critical for range queries!)
    print("\n" + "-" * 50)
    print("Step 1: Creating FLOAT indexes for numerical fields")
    print("(These are CRITICAL for Brix, pH, viscosity, etc. range queries)")
    print("-" * 50)
    
    for field in NUMERICAL_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f"numerical.{field}",
                field_schema=PayloadSchemaType.FLOAT,
                wait=True
            )
            print(f"  ✓ Created FLOAT index: numerical.{field}")
            created_count += 1
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  - Skipped (exists): numerical.{field}")
                skipped_count += 1
            else:
                print(f"  ✗ Failed: numerical.{field} - {e}")
    
    # 2. Create KEYWORD indexes for spec_fields (for exact match filtering)
    print("\n" + "-" * 50)
    print("Step 2: Creating KEYWORD indexes for spec_fields")
    print("-" * 50)
    
    for field in SPECIFIED_FIELDS:
        try:
            # Use KEYWORD for categorical fields, FLOAT for numerical
            if field in NUMERICAL_FIELDS:
                schema = PayloadSchemaType.FLOAT
            else:
                schema = PayloadSchemaType.KEYWORD
            
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f"spec_fields.{field}",
                field_schema=schema,
                wait=True
            )
            print(f"  ✓ Created {schema.name} index: spec_fields.{field}")
            created_count += 1
        except Exception as e:
            if "already exists" in str(e).lower():
                skipped_count += 1
            else:
                print(f"  ✗ Failed: spec_fields.{field} - {e}")
    
    # 3. Create metadata indexes
    print("\n" + "-" * 50)
    print("Step 3: Creating metadata indexes")
    print("-" * 50)
    
    metadata_indexes = [
        ("country", PayloadSchemaType.KEYWORD),
        ("version", PayloadSchemaType.KEYWORD),
        ("recipe_name", PayloadSchemaType.KEYWORD),
        ("num_available", PayloadSchemaType.INTEGER),
        ("num_missing", PayloadSchemaType.INTEGER),
    ]
    
    for field, schema in metadata_indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
                wait=True
            )
            print(f"  ✓ Created {schema.name} index: {field}")
            created_count += 1
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  - Skipped (exists): {field}")
                skipped_count += 1
            else:
                print(f"  ✗ Failed: {field} - {e}")
    
    # 4. Create text index for full-text search
    print("\n" + "-" * 50)
    print("Step 4: Creating full-text index")
    print("-" * 50)
    
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="description",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True
            ),
            wait=True
        )
        print("  ✓ Created TEXT index: description")
        created_count += 1
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  - Skipped (exists): description")
            skipped_count += 1
        else:
            print(f"  ✗ Failed: description - {e}")
    
    print("\n" + "=" * 70)
    print("INDEX CREATION COMPLETE")
    print(f"  Created: {created_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print("=" * 70)
    
    # 5. Optimize collection (optional but recommended)
    print("\n" + "-" * 50)
    print("Step 5: Triggering collection optimization")
    print("(This will consolidate segments and improve search performance)")
    print("-" * 50)
    
    try:
        # Update collection to ensure optimization happens
        client.update_collection(
            collection_name=collection_name,
            optimizer_config={
                "indexing_threshold": 10000,  # Trigger optimization after 10K points
            }
        )
        print("  ✓ Optimization triggered")
    except Exception as e:
        print(f"  - Could not trigger optimization: {e}")
    
    print("\n✅ DONE! Your searches should now be MUCH faster.")
    print("   Try the same query again in the UI.")
    
    return True


if __name__ == "__main__":
    # Allow overriding host via command line
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 6333
    
    create_payload_indexes(
        qdrant_host=host,
        qdrant_port=port
    )

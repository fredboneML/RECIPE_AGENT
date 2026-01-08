#!/bin/bash
# count_recipe_versions.sh - Counts all L and P version recipes in Qdrant

L_COUNT=0
P_COUNT=0
OTHER_COUNT=0
OFFSET=""
BATCH=0

echo "Counting all 639,912 recipes (this will take several minutes)..."

while true; do
  BATCH=$((BATCH + 1))
  
  # Build request - offset must be quoted as string for UUID/ID
  if [ -z "$OFFSET" ]; then
    REQUEST='{"limit": 10000, "with_payload": ["recipe_name"], "with_vector": false}'
  else
    REQUEST="{\"limit\": 10000, \"with_payload\": [\"recipe_name\"], \"with_vector\": false, \"offset\": \"$OFFSET\"}"
  fi
  
  # Make API call
  RESPONSE=$(curl -s -X POST http://localhost:6333/collections/food_recipes_two_step/points/scroll \
    -H "Content-Type: application/json" \
    -d "$REQUEST")
  
  # Extract recipe names
  NAMES=$(echo "$RESPONSE" | jq -r '.result.points[].payload.recipe_name // empty' 2>/dev/null)
  
  # Check if we got any results
  POINT_COUNT=$(echo "$RESPONSE" | jq '.result.points | length' 2>/dev/null)
  if [ "$POINT_COUNT" = "0" ] || [ -z "$POINT_COUNT" ]; then
    echo "No more points to process"
    break
  fi
  
  # Count versions in this batch
  BATCH_L=$(echo "$NAMES" | grep -c '_L$' 2>/dev/null || echo 0)
  BATCH_P=$(echo "$NAMES" | grep -c '_P$' 2>/dev/null || echo 0)
  BATCH_TOTAL=$(echo "$NAMES" | wc -l | tr -d ' ')
  BATCH_OTHER=$((BATCH_TOTAL - BATCH_L - BATCH_P))
  
  L_COUNT=$((L_COUNT + BATCH_L))
  P_COUNT=$((P_COUNT + BATCH_P))
  OTHER_COUNT=$((OTHER_COUNT + BATCH_OTHER))
  
  TOTAL=$((L_COUNT + P_COUNT + OTHER_COUNT))
  echo "Batch $BATCH ($POINT_COUNT pts): L=$BATCH_L P=$BATCH_P | Total so far: $TOTAL / 639912"
  
  # Get next offset (it's a point ID, could be UUID or integer)
  OFFSET=$(echo "$RESPONSE" | jq -r '.result.next_page_offset // empty' 2>/dev/null)
  
  if [ -z "$OFFSET" ] || [ "$OFFSET" = "null" ]; then
    echo "Reached end of collection"
    break
  fi
done

echo ""
echo "========================================"
echo "=== FINAL COUNT ==="
echo "Version L (Prototype): $L_COUNT"
echo "Version P (Production):  $P_COUNT"
echo "Other formats:          $OTHER_COUNT"
echo "----------------------------------------"
echo "TOTAL:                  $((L_COUNT + P_COUNT + OTHER_COUNT))"
echo "========================================"
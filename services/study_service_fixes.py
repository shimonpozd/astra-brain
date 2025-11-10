# Fixes for study_service.py

# Fix 1: Correct import and function call (around line 488-491)
            try:
                from config.personalities import get_personality
                personality_config = get_personality(agent_id)
                system_prompt = personality_config.get("system_prompt", "") if personality_config else ""
            except Exception as e:
                logger.warning(f"Failed to load personality {agent_id}: {e}")
                personality_config = {}
                system_prompt = ""

# Fix 2: Safe access to workbench attributes (around line 508-512)
            # Workbench panels - safe access since workbench might be a dict
            if snapshot.workbench:
                # Handle both dict and object access patterns
                if hasattr(snapshot.workbench, 'left'):
                    # Object-style access
                    if snapshot.workbench.left:
                        context_parts.append(f"Left Workbench: {snapshot.workbench.left}")
                    if snapshot.workbench.right:
                        context_parts.append(f"Right Workbench: {snapshot.workbench.right}")
                elif isinstance(snapshot.workbench, dict):
                    # Dict-style access
                    if snapshot.workbench.get('left'):
                        context_parts.append(f"Left Workbench: {snapshot.workbench['left']}")
                    if snapshot.workbench.get('right'):
                        context_parts.append(f"Right Workbench: {snapshot.workbench['right']}")
                else:
                    logger.warning(f"Unexpected workbench type: {type(snapshot.workbench)}")

# Alternative simpler fix for workbench (if we know it's always a dict):
            if snapshot.workbench:
                left_ref = snapshot.workbench.get('left') if isinstance(snapshot.workbench, dict) else getattr(snapshot.workbench, 'left', None)
                right_ref = snapshot.workbench.get('right') if isinstance(snapshot.workbench, dict) else getattr(snapshot.workbench, 'right', None)
                
                if left_ref:
                    context_parts.append(f"Left Workbench: {left_ref}")
                if right_ref:
                    context_parts.append(f"Right Workbench: {right_ref}")































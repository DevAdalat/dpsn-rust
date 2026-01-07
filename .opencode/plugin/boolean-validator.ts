import type { Plugin } from "@opencode-ai/plugin"

export const EditBooleanFix: Plugin = async ({ $, client }) => {
  let hadTypeConversion = false;
  let convertedFields: string[] = [];

  return {
    "tool.execute.before": async (input, output) => {
      // Reset tracking for each execution
      hadTypeConversion = false;
      convertedFields = [];

      // Intercept edit tool calls
      if (input.tool === "edit") {
        // Transform string booleans to actual booleans ONLY for replaceAll
        const args = output.args;

        // Only check the replaceAll property
        if (args.hasOwnProperty('replaceAll') && typeof args.replaceAll === 'string') {
          // Check for "true" or "false" strings
          if (args.replaceAll === "true") {
            args.replaceAll = true;
            hadTypeConversion = true;
            convertedFields.push(`replaceAll: "true" → true`);
          } else if (args.replaceAll === "false") {
            args.replaceAll = false;
            hadTypeConversion = true;
            convertedFields.push(`replaceAll: "false" → false`);
          }
        }
      }
    },
    
    "tool.execute.after": async (input, output) => {
      // Return warning message if type conversion happened
      if (input.tool === "edit" && hadTypeConversion) {
        const warningMessage = `⚠️ STRICT WARNING: Wrong prop type detected and auto-corrected!\n\nConverted fields:\n${convertedFields.map(f => `  - ${f}`).join('\n')}\n\nPlease use boolean values (true/false) instead of strings ("true"/"false") for the replaceAll parameter.`;
        
        // Append warning to the tool result so AI sees it
        if (typeof output === 'object' && output !== null) {
          output.warning = warningMessage;
        }
        
        // Return modified output with warning
        return {
          ...output,
          _pluginWarning: warningMessage
        };
      }
    },
  }
}

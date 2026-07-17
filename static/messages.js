// Every bit of status text, confirmation, and error message the dashboard
// shows lives here in one place, instead of being copy-pasted everywhere.

const MESSAGES = {
  saving: "Saving...",
  checking: "Checking...",
  starting: "Starting...",
  idle: "Idle.",
  noFileSelected: "No file selected.",
  noFileToPreview: "No file to preview.",
  noPairsInFile: "No Q&A pairs in this file yet.",
  noPairsOrTextInFolder: "No Q&A pairs or text found in this folder.",
  questionAndAnswerRequired: "Both question and answer are required.",
  chooseFileFirst: "Choose a file first (Browse).",
  enterDestinationPath: "Enter a destination path.",
  confirmDeletePair: "Delete this Q&A pair? This can't be undone.",
  micDenied: "Microphone access was denied or is unavailable.",
  noSpeechCaught: "Didn't catch any speech -- please try again.",
  speechSynthesisFailed: "Speech synthesis failed.",
  generationStoppedEarly: "Generation stopped early.",
  usingDefaultTranslateModel: "Using default per-language translation model.",
  chatKeyRequired: "Enter a key to start chatting.",
  unlocking: "Unlocking...",
  unlocked: "Unlocked.",
  confirmClearChat: "Clear the chat? This can't be undone, and you'll need a new key to start again.",
};

// Turns a count into "1 pair" / "2 pairs" so that pluralization logic isn't
// copy-pasted at every place a count gets shown to you.
function pluralize(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

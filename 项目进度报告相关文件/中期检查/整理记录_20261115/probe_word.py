import win32com.client
app=None
try:
 app=win32com.client.DispatchEx('Word.Application');app.Visible=False;app.DisplayAlerts=0
 print('WORD_RENDERER_AVAILABLE',app.Version)
finally:
 if app is not None:app.Quit()

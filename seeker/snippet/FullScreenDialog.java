//date: 2022-03-22T17:02:34Z
//url: https://api.github.com/gists/f9970f40406e17817c7c52f344731ced
//owner: https://api.github.com/users/virendersran01

public class FullScreenDialog extends DialogFragment {
                                  
    public static final String TAG = "FullScreenDialog";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setStyle(DialogFragment.STYLE_NORMAL, R.style.FullScreenDialog);
    }

    @Override
    public void onStart() {
        super.onStart();

        Dialog dialog = getDialog();
        if (dialog != null) {
            int width = ViewGroup.LayoutParams.MATCH_PARENT;
            int height = ViewGroup.LayoutParams.MATCH_PARENT;
            dialog.getWindow().setLayout(width, height);
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup parent, Bundle state) {
        super.onCreateView(inflater, parent, state);

        View view = getActivity().getLayoutInflater().inflate(R.layout.dialog_layout, parent, false);
        return view;
    }
}
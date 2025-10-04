import pandas as pd 
import numpy as np
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

def analyze_document(pdf_path, endpoint, api_key, output_excel):
    """
    Extract tables from a PDF using Microsoft Document Intelligence (Form Recognizer)
    and save each table into its own sheet in an Excel workbook.

    Steps for each table:
      1) Build a DataFrame from the detected cells.
      2) Remove rows with fewer than 2 non-empty cells.
      3) Perform a horizontal fill from left to right.
      4) Remove columns that are entirely empty.
      5) Transpose so the first column becomes the header row (if lengths match).
      6) Remove empty columns again.
      7) Write to Excel.
    """
    # Initialize the Document Analysis client
    client = DocumentAnalysisClient(
        endpoint=endpoint, 
        credential=AzureKeyCredential(api_key)
    )

    # Analyze the PDF with the prebuilt-document model
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-document", document=f)
        result = poller.result()

    all_tables = []
    table_counter = 1

    for table in result.tables:
        # Determine the size of the table
        max_row = max(cell.row_index for cell in table.cells) + 1
        max_col = max(cell.column_index for cell in table.cells) + 1

        # Build an empty matrix for the table
        matrix = [["" for _ in range(max_col)] for _ in range(max_row)]
        for cell in table.cells:
            matrix[cell.row_index][cell.column_index] = cell.content

        # Convert to DataFrame
        df = pd.DataFrame(matrix)

        # ---------------------------------------------------------------
        # 1) Remove rows with fewer than 2 non-empty cells
        # ---------------------------------------------------------------
        def count_nonempty_cells(row):
            return sum(1 for val in row if val.strip() != "")
        df = df[ df.apply(count_nonempty_cells, axis=1) >= 2 ].copy()
        df.reset_index(drop=True, inplace=True)

        # If the table is now empty, skip further processing
        if df.empty:
            continue

        # ---------------------------------------------------------------
        # 2) Horizontal fill from left to right
        #    If cell j is empty, fill it with cell j-1 (if non-empty)
        # ---------------------------------------------------------------
        for i in range(len(df)):
            for j in range(1, df.shape[1]):
                if df.iat[i, j].strip() == "" and df.iat[i, j-1].strip() != "":
                    df.iat[i, j] = df.iat[i, j-1]

        # ---------------------------------------------------------------
        # 3) Remove columns that are entirely empty
        # ---------------------------------------------------------------
        df = df.loc[:, df.apply(lambda col: any(cell.strip() != "" for cell in col))]

        # ---------------------------------------------------------------
        # 4) Transpose so the first column becomes the header row
        # ---------------------------------------------------------------
        if not df.empty and df.shape[1] > 1:
            df_t = df.transpose().reset_index(drop=True)

            if not df_t.empty:
                # The first row of df_t is our potential header
                new_header = df_t.iloc[0].tolist()
                # Check if header length matches the number of columns
                if len(new_header) == df_t.shape[1]:
                    df_t = df_t[1:].reset_index(drop=True)
                    df_t.columns = new_header
                    df = df_t
                else:
                    df = df_t

        # ---------------------------------------------------------------
        # 5) Remove empty columns again after transposition
        # ---------------------------------------------------------------
        if not df.empty:
            df = df.loc[:, df.apply(lambda col: any(cell.strip() != "" for cell in col))]

        all_tables.append((table_counter, df))
        table_counter += 1

    # Write each table to a separate sheet in the Excel file
    with pd.ExcelWriter(output_excel) as writer:
        for num, df in all_tables:
            sheet_name = f"Table_{num}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Extraction complete. {len(all_tables)} table(s) saved to '{output_excel}'.")


def combine_sheets_order(input_excel, output_excel):
    """
    Combines all sheets from an Excel file into a single sheet. The order of columns
    is determined by:
      - The columns in the first sheet (in order), followed by
      - Any extra columns from the remaining sheets (in the order they first appear)
    
    Missing values in sheets that don't contain a particular column will be NaN.
    
    :param input_excel: Path to the input Excel file with multiple sheets.
    :param output_excel: Path to the final output Excel file (single sheet).
    """
    # Read all sheets (pandas returns an OrderedDict preserving sheet order)
    sheets_dict = pd.read_excel(input_excel, sheet_name=None)
    
    if not sheets_dict:
        print("No sheets found in the input Excel file.")
        return
    
    # Get the list of sheet names in order
    sheet_names = list(sheets_dict.keys())
    
    # Use the first sheet's columns as the initial column order
    combined_columns = list(sheets_dict[sheet_names[0]].columns)
    
    # For each sheet (in order), add any extra columns not already in combined_columns.
    for sheet_name, df in sheets_dict.items():
        for col in df.columns:
            if col not in combined_columns:
                combined_columns.append(col)
    
    # Reindex each DataFrame to have all columns in the determined order.
    dfs = []
    for sheet_name, df in sheets_dict.items():
        df_reindexed = df.reindex(columns=combined_columns)
        dfs.append(df_reindexed)
    
    # Concatenate all sheets row-wise.
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Write the combined DataFrame to a single sheet in the output Excel file.
    combined_df.to_excel(output_excel, index=False)
    print(f"Combined sheets have been written to '{output_excel}'.")


def main():
    # Set your Document Intelligence credentials and file paths here
    endpoint = ""          # Your Document Intelligence endpoint
    api_key = ""    # Your Document Intelligence API key
    pdf_path = ""  # Path to your PDF file
    multi_sheet_excel = ""   # Temporary Excel file with multiple sheets
    final_output_excel = "" # Final Excel file with a single sheet

    # Step 1: Extract tables from the PDF into a multi-sheet Excel file
    analyze_document(pdf_path, endpoint, api_key, multi_sheet_excel)
    
    # Step 2: Combine all sheets from the multi-sheet Excel into a single sheet
    combine_sheets_order(multi_sheet_excel, final_output_excel)


if __name__ == "__main__":
    main()
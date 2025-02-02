import re
from abc import ABC
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ALLOWED_RAM_VALUES = {1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 256, 512, 1024}
COLORS_RU_TO_EN = {
    "зелёный": "green",
    "черный": "black",
    "розовое золото": "rose gold",
    "серебро": "silver",
    "коричневый": "brown",
    "голубой": "light blue",
    "розовый": "pink",
    "серый космос": "space gray",
    "желтый": "yellow",
    "сиреневый": "lilac",
    "золотой": "gold",
    "синий": "blue",
    "фиолетовый": "purple",
    "серебристый": "silver",
    "золото": "gold",
    "красный": "red",
    "оранжевый": "orange",
    "темно-синий": "dark blue",
    "зеленый": "green",
    "коралловый": "coral",
    "белый": "white",
    "бежевый": "beige",
    "cерый": "gray",
    "серый": "gray",
}
CONVERT_ROM_TO_GB = {
    "1": "1024",
    "2": "2048",
    "3": "3072",
    "4": "4096",
    "6": "6144",
    "8": "8192",
}


def get_normalyze_color(color_en, color_ru):
    if pd.isna(color_en) and not pd.isna(color_ru):
        return COLORS_RU_TO_EN[color_ru.lower()]
    return color_en


def get_color(name):
    match = re.search(
        r"\b("
        r"Cyclone Transparent Black|Bright Silver Edition|Awesome Lime Yellow|Seven Degree Purple|Obsidian Midnight|Interstellar Blue|"
        r"Interstellar Grey|Sandstone Orange|Moonlight Silver|Multicolor Green|Mercurial Silver|Natural Titanium|Awesome Graphite|"
        r"Magic Skin Green|Deep Ocean Black|Impression Green|Gradation Purple|Midnight Shadow|Gradient Bronze|Titanium Yellow|"
        r"Celestial Black|Titanium Silver|Moonlight White|Moving Titanium|Startrail Black|Magic Skin Blue|Dreamland Green|Titanium Orange|"
        r"Gradation Black|Platinum Silver|Lavender Purple|Metaverse Green|Gradation Green|Desert Titanium|Awesome Iceblue|Titanium Violet|"
        r"Ice Mirror Blue|Silver Fantasy|Digital Silver|Meteorite Grey|Sparkle Purple|Crystal Silver|Titanium Green|Uranolith Gray|"
        r"Moonlight Blue|Chromatic Gray|Titanium Black|Turquoise Cyan|Metallic Black|Black Titanium|Black Burgundy|Alpenglow Gold|Midnight Black|"
        r"Nighttime Blue|Awesome Silver|Celadon Marble|Glittery White|Obsidian Black|Awesome Violet|Stargase White|Stellar Shadow|Hurricane Blue|"
        r"Sunrise Orange|Ceramics White|White Titanium|Infinite Black|Phantom Silver|Deepsea Luster|Stellar Black|Titanium Gold|Awesome Lilac|"
        r"Titanium Grey|Phantom Black|Dark Illusion|Ceramic White|Graphite Gray|Starry Silver|Glowing Green|Fluorite Blue|Cobalt Violet|Rutile Orange|"
        r"Glacier White|Gravity Black|Crystal Green|Mystery White|Glowing Black|Dreamy Purple|Deep Sea Blue|Basaltic Dark|Cryolite Blue|Atlantic Blue|"
        r"Titanium Gray|Emerald Green|Midnight Gray|Nebula Violet|Flowy Emerald|Dual Jade Fog|Memphis Green|Variable Gold|Daybreak Blue|Silver Shadow|"
        r"Sunset Melody|Stardust Grey|Iceland White|Awesome Black|Twilight Blue|Awesome White|Blue Titanium|Eternal Black|Champion Gold|Crystal Black|"
        r"Sapphire Blue|Seasalt White|Grey Graphite|Obsidian Edge|Starlit Black|Awesome Lemon|Phantom White|Hurrican Blue|Violet Garden|Titanium Blue|"
        r"Awesome Peach|Mecha Silver|Glacier Glow|Diamond Grey|Skyline Blue|Amber Yellow|Mecha Orange|Aurora Green|Mirror Black|Cosmos Black|Frost Silver|"
        r"Crystal Blue|Magma Orange|Thunder Grey|Aurora Cloud|Kyanite Blue|Monet Purple|Forest Green|Awesome Navy|Voyager Grey|Morning Mist|Awesome Blue|"
        r"Horizon Gold|Stormy Black|Snowy Silver|Meteor Black|Starlit Blue|Mighty Black|Frosty Ivory|Light Silver|Polar Silver|Black Aurora|Blossom Glow|"
        r"Magnet Black|Awesome Lime|Pacific Blue|Viva Magenta|Awesome Mint|Cloudy White|Rococo Pearl|Alpine Green|Glacier Blue|Carbon Black|Awesome Gray|"
        r"Starry Black|Glowing Gold|Meadow Green|Ultramarine|Alpine Blue|Polar Black|Prism Black|Wintergreen|Starry Blue|Oasis Green|Kinda Coral|"
        r"Khaki Green|Sunset Gold|Power Black|Comet Green|Clash White|Black Stone|Night Black|Space Black|Pastel Lime|Mars Orange|Sleek Black|Green Field|"
        r"Galaxy Grey|Cross Black|Magic Black|Dark Chrome|Mecha Black|Paper White|Iris Purple|Lemon Green|Marble Gray|Sierra Blue|Rose Quartz|Dark Purple|"
        r"Dark Matter|Deep Purple|Light Green|Sorta Sunny|Dark Welkin|Nebula Blue|Bora Purple|Shadow Grey|Pearl White|Arctic Glow|Desert Sand|Titan Black|"
        r"Lava Orange|Coral Green|Mirror Grey|Bahama Blue|Green Oasis|Speed Green|Speed Black|Mint Green|Rebel Grey|Storm Gray|Light Grey|Ice Silver|"
        r"Speed Blue|Rush Black|Cyber Blue|Misty Aqua|Cross Blue|Light Gray|Snapdragon|Rock Black|Water Gray|Space Grey|Lemongrass|Rome Green|Mist Black|"
        r"Pearl Gold|Onyx Black|Gray Green|Cocoa Gold|Pine Green|Sandy Gold|Blue Oasis|Agua Green|Cloud Mint|Titan Blue|Jade Green|Misty Grey|Blue Black|"
        r"Rangi Blue|Just Black|Navy Black|CobaltBlue|Titan Gray|Mecha Blue|Jewel Blue|Ocean Blue|Amber Gold|Space Gray|Prism Blue|Ocean Teal|Black Onyx|"
        r"Dark Green|Beige Sand|Sage Green|Grey/green|Cloud Navy|Light Blue|Water Blue|Iron Gray|SlateGray|Azure Sky|Sea Green|Lake Blue|Stainless|"
        r"Palm Blue|Star Blue|Aqua Blue|Turquoise|Cool Blue|Dawn Gold|Starlight|Porcelain|Tangerine|Eco Black|Neon Gold|Pink Gold|Deep Grey|Cyan Lake|"
        r"Rose Gold|Jet Black|Dark Blue|Sky Cyan|Icy Blue|Graphite|Burgundy|Platinum|Amethyst|Lavender|Sapphire|Blue Sea|Dark Red|Titanium|Midnight|"
        r"Obsidian|Charcoal|Ice Blue|Sea Blue|Basaltic|Mondrian|Emerald|Iceblue|Natural|Crimson|Silver|Violet|Orange|Indigo|Yellow|Maroon|Copper|Purple|"
        r"Desert|White|Lilac|Brown|Lemon|Peony|Pearl|Black|Hazel|Peach|Coral|Green|Olive|Beige|Azure|Cream|Blue|Cyan|Aloe|Gray|Aqua|Milk|Gold|Grey|Snow|"
        r"Lime|Navy|Teal|Mint|Rose|Pink|Glow|Sea|Bay|Red"
        r")\b",
        name,
        re.IGNORECASE,
    )
    return match.group(0) if match else None


def get_manufacturer(name):
    match = re.search(
        r"\b("
        r"OnePlus|Apple|Samsung|Infinix|Sony|Itel|ZTE|Vivo|Huawei|"
        r"Nothing|Blackview|MOTOROLA|Google|ASUS|Microsoft|"
        r"Oppo|Honor|Poco|Asus|Realme|Tecno|Xiaomi|POCO"
        r")\b",
        name,
        re.IGNORECASE,
    )
    return match.group(0) if match else None


def clean_text(text):
    text = re.sub(r"[^\w\s/.+-]", "", str(text)).strip()
    return text


def get_ram(text):
    match = re.findall(
        r"(\d+)(?:\+|/|\s)(?=\d{1,4}[TtGgBb]|\d{2,4}\b)", text, re.IGNORECASE
    )
    if not match:
        return None
    ram = match[-1]
    return ram if int(ram) in ALLOWED_RAM_VALUES else None


def get_rom(text):
    match = re.search(r"(?:\d+\+)?(32|64|128|256|512|1024)[GgBb ]", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(?:\d+\+)?(1|2|3|4)(Tb|TB|T|t)", text, re.IGNORECASE)
    return match.group(1) if match else None


def get_price(text):
    price_matches = re.findall(r"(\b\d{4,6})\s*[₽$€]?", str(text))
    if not price_matches:
        return None
    last_price = int(price_matches[-1])
    return last_price if last_price >= 2025 else None


class CollectData(ABC):
    def __init__(self, file: pd.DataFrame) -> None:
        self._file = file
        self.data = self._collect_data()
        self._collect_manufacturer()

    def _collect_data(self) -> pd.DataFrame: ...

    def _collect_manufacturer(self) -> None:
        self.data["Производитель"] = self.data["Наименование товара"].apply(
            get_manufacturer
        )

    def get_data(self) -> pd.DataFrame:
        return self.data


class MiHonor(CollectData):
    def __init__(self, file: pd.DataFrame) -> None:
        super().__init__(file)

    @staticmethod
    def _remove_specifications(text: str, color: str, ram: str, rom: str) -> str:
        if color:
            text = text.replace(color, "").strip()
        if ram and rom:
            pattern = rf"{ram}\s*[+/]?\s*{rom}\s*[GgBb]"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        elif ram:
            pattern = rf"{ram}\s*[GgBb]"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        elif rom:
            pattern = rf"{rom}\s*[GgBb]"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        if ram or rom:
            pattern = r"\b[A-Za-z]\b"
            text = re.sub(pattern, "", text).strip()
        return re.sub(r"\s+", " ", text).strip()

    def _collect_data(self) -> pd.DataFrame:
        file = self._file
        file.dropna(inplace=True)
        MiHonor = file[file["поставщик"] == "MiHonor"].copy()
        MiHonor["Цена"] = MiHonor["прайс"].apply(get_price)
        MiHonor["price_clean"] = MiHonor["прайс"].apply(
            lambda x: re.sub(r"\s\d+$", "", clean_text(x)).strip()
        )
        MiHonor["Цвет"] = MiHonor["price_clean"].apply(get_color)
        MiHonor["RAM"] = MiHonor["price_clean"].apply(get_ram)
        MiHonor["ROM"] = MiHonor["price_clean"].apply(get_rom)
        MiHonor["Наименование товара"] = MiHonor.apply(
            lambda row: self._remove_specifications(
                row["price_clean"], row["Цвет"], row["RAM"], row["ROM"]
            ),
            axis=1,
        )
        col = MiHonor.pop("Наименование товара")
        MiHonor.insert(1, "Наименование товара", col)
        MiHonor = MiHonor[MiHonor["price_clean"].str.strip() != ""]
        MiHonor = MiHonor.dropna(subset=["Цена", "Цвет"], how="all")
        MiHonor = MiHonor.dropna(subset=["Цвет", "RAM", "ROM"], how="all")
        return MiHonor


class Hi(CollectData):
    def __init__(self, file: pd.DataFrame) -> None:
        super().__init__(file)

    @staticmethod
    def _clean_text_price(text: str) -> str:
        text = re.sub(r"[^\w\s/.+-]", "", text).strip()
        return re.sub(r"\s\d+$", "", text).strip()

    @staticmethod
    def _clean_text_final(text: str, color: str, rom: str) -> str:
        if rom:
            text = text.replace(rom, "").strip()
        if color:
            text = text.replace(color, "").strip()
        return re.sub(r"\b(?:GB|TB|T|G)\b(?!\d)", "", text, flags=re.IGNORECASE).strip()

    def _collect_data(self) -> pd.DataFrame:
        file = self._file
        file.dropna(inplace=True)
        Hi = file[file["поставщик"] == "HI"].copy()
        Hi["price_clean"] = Hi["прайс"].apply(clean_text)
        Hi["Цена"] = Hi["price_clean"].apply(get_price)
        Hi["ROM"] = Hi["price_clean"].apply(get_rom)
        Hi["Наименование товара"] = Hi["price_clean"].apply(self._clean_text_price)
        Hi["Цвет"] = Hi["Наименование товара"].apply(get_color)
        Hi["Наименование товара"] = Hi.apply(
            lambda row: self._clean_text_final(
                row["Наименование товара"], row["Цвет"], row["ROM"]
            ),
            axis=1,
        )
        Hi = Hi[Hi["price_clean"].str.strip() != ""]
        Hi = Hi.dropna(subset=["Цена"], how="all")
        return Hi


class YouTakeAll(CollectData):
    def __init__(self, file: pd.DataFrame) -> None:
        super().__init__(file)

    @staticmethod
    def _remove_specifications(text: str, color: str, ram: str, rom: str) -> str:
        if color:
            text = text.replace(color, "").strip()
        if rom:
            text = re.sub(rom, "", text, flags=re.IGNORECASE).strip()
        if ram:
            text = re.sub(rf"(?:({ram}/))", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+", " ", text).strip()
        return re.sub(r"\b(?:GB|TB|T|G)\b(?!\d)", "", text, flags=re.IGNORECASE).strip()

    def _collect_data(self) -> pd.DataFrame:
        file = self._file
        file.dropna(inplace=True)
        YouTakeAll = file[file["поставщик"] == "YouTakeAll"].copy()
        YouTakeAll["Цена"] = YouTakeAll["прайс"].apply(get_price)
        YouTakeAll["price_clean"] = YouTakeAll["прайс"].apply(
            lambda x: re.sub(
                r"\s\d+$", "", clean_text(re.sub(r"[а-яА-ЯёЁ]", "", x).strip())
            ).strip()
        )
        YouTakeAll["RAM"] = YouTakeAll["price_clean"].apply(get_ram)
        YouTakeAll["ROM"] = YouTakeAll["price_clean"].apply(get_rom)
        YouTakeAll["Цвет"] = YouTakeAll["price_clean"].apply(get_color)
        YouTakeAll["Наименование товара"] = YouTakeAll.apply(
            lambda row: self._remove_specifications(
                row["price_clean"], row["Цвет"], row["RAM"], row["ROM"]
            ),
            axis=1,
        )
        col = YouTakeAll.pop("Наименование товара")
        YouTakeAll.insert(1, "Наименование товара", col)
        YouTakeAll = YouTakeAll.dropna(subset=["Цена", "RAM", "ROM", "Цвет"], how="all")
        return YouTakeAll


class Pav112(CollectData):
    def __init__(self, file: pd.DataFrame) -> None:
        super().__init__(file)

    @staticmethod
    def _remove_specifications(
        text: str, color: str, ram: str, rom: str, price: str
    ) -> str:
        if price and pd.notna(price):
            text = text.replace(str(int(price)), "").strip()
        if color:
            text = text.replace(color, "").strip()
        if rom:
            text = re.sub(rom, "", text, flags=re.IGNORECASE).strip()
        if ram:
            text = re.sub(rf"(?:({ram}/))", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\b(?:GB|TB|T|G)\b(?!\d)", "", text, flags=re.IGNORECASE).strip()
        return re.sub(r"[\-.]$", "", text).strip()

    def _collect_data(self) -> pd.DataFrame:
        file = self._file
        file.dropna(inplace=True)
        pav112 = file[file["поставщик"] == "112пав"].copy()
        pav112["price_clean"] = pav112["прайс"].apply(
            lambda x: re.sub(r"\.$", "", clean_text(x))
        )
        pav112["Цена"] = pav112["price_clean"].apply(get_price)
        pav112["RAM"] = pav112["price_clean"].apply(get_ram)
        pav112["ROM"] = pav112["price_clean"].apply(get_rom)
        pav112["Цвет"] = pav112["price_clean"].apply(get_color)
        pav112["Наименование товара"] = pav112.apply(
            lambda row: self._remove_specifications(
                row["price_clean"], row["Цвет"], row["RAM"], row["ROM"], row["Цена"]
            ),
            axis=1,
        )
        col = pav112.pop("Наименование товара")
        pav112.insert(1, "Наименование товара", col)
        pav112 = pav112.dropna(subset=["Цена", "RAM", "ROM", "Цвет"], how="all")
        return pav112


def standardize_memory(value):
    if pd.isna(value):
        return None
    if value in CONVERT_ROM_TO_GB:
        return f"{CONVERT_ROM_TO_GB[value]}GB"
    return f"{value}GB"


def make_comparison(row) -> str:
    name = row.get("Наименование товара")
    ram = row.get("RAM")
    rom = row.get("ROM")
    color = row.get("Цвет")
    is_ram_na = pd.isna(ram)
    is_rom_na = pd.isna(rom)
    is_color_na = pd.isna(color)
    if not name:
        return None
    if not is_color_na:
        color = " ".join(word.capitalize() for word in str(color).split())
    if is_ram_na and not is_rom_na and not is_color_na:
        return f"{name} {rom} ({color})"
    if is_color_na and not is_ram_na and not is_rom_na:
        return f"{name} {ram}/{rom}"
    if is_ram_na and is_color_na and not is_rom_na:
        return f"{name} {rom}"
    if is_ram_na and is_color_na and is_rom_na:
        return name
    if is_ram_na and not is_color_na and is_rom_na:
        return f"{name} ({color})"
    return f"{name} {ram}/{rom} ({color})"


def get_normalyze(text, color_en, ram, rom, type_=None):
    if pd.isna(text):
        return text
    if type_:
        text = text.replace(str(type_), "").strip()
    if pd.isna(ram):
        text += f" | color: {color_en}, ram: {ram}, rom: {rom}"
    else:
        text += f" | color: {color_en}, ram: {int(ram)}, rom: {rom}"
    return re.sub(r"\s+", " ", str(text).strip().lower())


def get_df_manufacturer(dfs: List[CollectData]) -> pd.DataFrame:
    all_data = pd.concat([df.data for df in dfs], ignore_index=True)
    all_data["RAM"] = all_data["RAM"].astype("Int64")
    all_data["ROM"] = all_data["ROM"].apply(standardize_memory)
    all_data["Сопоставление"] = all_data.apply(make_comparison, axis=1)
    all_data["Сопоставление"] = all_data.apply(
        lambda row: get_normalyze(
            text=row["Сопоставление"],
            color_en=row["Цвет"],
            ram=row["RAM"],
            rom=row["ROM"],
        ),
        axis=1,
    )
    return all_data


def get_normalyze_df_shop(df: pd.DataFrame) -> pd.DataFrame:
    memory_mapping = {
        "1024GB": ["1TB", "1ТБ"],
        "2048GB": ["2TB", "2ТБ"],
    }

    def standardize_memory(value, name):
        if pd.isna(value):
            rom = get_rom(name)
            return "1024GB" if rom == "1" else f"{rom}GB"

        for standard, variants in memory_mapping.items():
            if str(value) in variants:
                return standard
        return value

    df_shop = df.copy()
    df_shop["Встроенная память"] = df_shop.apply(
        lambda row: standardize_memory(row["Встроенная память"], row["Наименование"]),
        axis=1,
    )

    color = df_shop["Наименование"].apply(get_color)
    df_shop.insert(1, "color_en", color)
    df_shop["color_en"] = df_shop["color_en"].apply(lambda x: x.lower() if x else None)
    df_shop["color_en"] = df_shop.apply(
        lambda row: get_normalyze_color(
            color_en=row["color_en"],
            color_ru=row["Цвет"],
        ),
        axis=1,
    )

    df_shop["Сопоставление"] = df_shop.apply(
        lambda row: get_normalyze(
            text=row["Наименование"],
            type_=row["Тип аппарата"],
            color_en=row["color_en"],
            ram=row["Оперативная память (Gb)"],
            rom=row["Встроенная память"],
        ),
        axis=1,
    )

    return df_shop


def main() -> None:
    file = pd.read_excel("Прайсы с телеграма 28.01.xlsx")
    miHonor = MiHonor(file)
    hi = Hi(file)
    youTakeAll = YouTakeAll(file)
    pav112 = Pav112(file)

    df_manufacturer = get_df_manufacturer([miHonor, hi, youTakeAll, pav112])

    df_shop = pd.read_excel("Товары магазина.xlsx")
    df_shop = get_normalyze_df_shop(df_shop)

    model = SentenceTransformer("fine_tuned_mpnet_v3")

    new_df = pd.DataFrame(columns=["Товар поставщика", "Товар в магазине"])

    SIMILARITY = 0.9

    print("[WAITING] Происходит загрузка модели")
    embeddings = model.encode(
        df_shop["Сопоставление"].tolist(), normalize_embeddings=True
    )
    print("[MODEL DOWNLOAD]")

    for index, row in df_manufacturer.iterrows():
        if pd.notna(row["Сопоставление"]):
            query_embedding = model.encode(
                [row["Сопоставление"]], normalize_embeddings=True
            )
            scores = np.dot(embeddings, query_embedding.T).flatten()

            best_match_ind = np.argmax(scores)
            best_match_score = scores[best_match_ind]

            if best_match_score >= SIMILARITY:
                best_match = df_shop.iloc[best_match_ind]["Наименование"]

                new_row = {
                    "Товар в магазине": best_match,
                    "Поставщик": row["поставщик"],
                    "Цена": row["Цена"],
                    "Товар поставщика": row["прайс"],
                    "Сходство": best_match_score,
                }
                new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

        print(
            f"[{index + 1}/{len(df_manufacturer)}] - Обработка товара: {row['Сопоставление']}"
        )

    grouped = new_df.groupby("Товар в магазине")

    excel_data = []

    for product, group in grouped:
        row = {"Наше название": product}
        for i, (_, item) in enumerate(group.iterrows(), start=1):
            row[f"цена {i}"] = item["Цена"]
            row[f"поставщик {i}"] = item["Поставщик"]

        excel_data.append(row)

    df_excel = pd.DataFrame(excel_data)

    df_excel.to_excel("output_prices.xlsx", index=False)

    print("✅ Excel-файл успешно создан: output_prices.xlsx")


if __name__ == "__main__":
    main()
